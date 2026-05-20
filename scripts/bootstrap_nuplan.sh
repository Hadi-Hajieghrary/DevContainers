#!/usr/bin/env bash
set -euo pipefail

# Configurable nuPlan bootstrapper.
# It intentionally separates planner-essential data from raw sensors.
# Planner-essential data = maps + SQLite log DBs.
# Raw camera/LiDAR blobs are optional and disabled by default.

PROFILE="${NUPLAN_DATA_PROFILE:-planner_mini}"
FORCE="false"
PRINT_ONLY="false"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
URLS_FILE="${REPO_ROOT}/configs/dataset_urls.env"
PROFILE_DIR="${REPO_ROOT}/configs/data_profiles"

usage() {
  cat <<USAGE
Usage:
  $0 [--profile NAME_OR_FILE] [--force] [--print-plan]
     [--maps-only] [--no-download] [--with-sensors]
     [--max-active-mini-dbs N|all] [--trim-mode none|disable|delete]

Profiles in configs/data_profiles:
  planner_mini        maps + mini log DBs; no camera/LiDAR. Recommended.
  planner_mini_small  maps + mini log DBs; active subset metadata for 3 DBs.
  maps_only           maps only.
  mini_with_sensors   maps + mini DBs + camera/LiDAR blobs.
  no_download         create/check folders only.
  custom              edit configs/data_profiles/custom.env.

Important:
  Ego pose/state, agent tracks, traffic-light status, route/mission metadata,
  and scenario metadata are bundled inside the SQLite log DBs. They are not
  separate official downloads. Maps are separate. Raw camera/LiDAR blobs are
  separate and optional.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --force) FORCE="true"; shift ;;
    --print-plan) PRINT_ONLY="true"; shift ;;
    --maps-only) PROFILE="maps_only"; shift ;;
    --no-download) PROFILE="no_download"; shift ;;
    --with-sensors) PROFILE="mini_with_sensors"; shift ;;
    --max-active-mini-dbs) CLI_MAX_ACTIVE_MINI_DBS="$2"; shift 2 ;;
    --trim-mode) CLI_TRIM_MODE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -f "${URLS_FILE}" ]]; then
  echo "Missing URL manifest: ${URLS_FILE}" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "${URLS_FILE}"

PROFILE_FILE="${PROFILE}"
if [[ ! -f "${PROFILE_FILE}" ]]; then
  PROFILE_FILE="${PROFILE_DIR}/${PROFILE}.env"
fi
if [[ ! -f "${PROFILE_FILE}" ]]; then
  echo "Unknown profile '${PROFILE}'. Expected a file or one of:" >&2
  find "${PROFILE_DIR}" -maxdepth 1 -name '*.env' -printf '  - %f\n' | sed 's/.env$//' >&2
  exit 2
fi
# shellcheck source=/dev/null
source "${PROFILE_FILE}"

# CLI overrides after loading profile.
if [[ -n "${CLI_MAX_ACTIVE_MINI_DBS:-}" ]]; then MAX_ACTIVE_MINI_DBS="${CLI_MAX_ACTIVE_MINI_DBS}"; fi
if [[ -n "${CLI_TRIM_MODE:-}" ]]; then TRIM_MODE="${CLI_TRIM_MODE}"; fi

DATA_ROOT="${NUPLAN_DATA_ROOT:-/workspace/data}"
MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${DATA_ROOT}/maps}"
EXP_ROOT="${NUPLAN_EXP_ROOT:-/workspace/exp}"
ARCHIVE_DIR="${DATA_ROOT}/archives"
MINI_DIR="${DATA_ROOT}/nuplan-v1.1/splits/mini"
LEGACY_CACHE_MINI_DIR="${DATA_ROOT}/data/cache/mini"
MINI_DISABLED_DIR="${DATA_ROOT}/nuplan-v1.1/splits/mini.disabled"
ACTIVE_MANIFEST="${DATA_ROOT}/nuplan-v1.1/splits/mini_active_manifest.txt"
SENSOR_DIR="${DATA_ROOT}/nuplan-v1.1/sensor_blobs"

mkdir -p "${DATA_ROOT}" "${MAPS_ROOT}" "${EXP_ROOT}" "${ARCHIVE_DIR}" "${MINI_DIR}" "${SENSOR_DIR}"

normalize_mini_dir() {
  # Some nuPlan mini archives unpack into data/cache/mini. Move DBs into the
  # canonical planner path so downstream tools always read one location.
  if [[ ! -d "${LEGACY_CACHE_MINI_DIR}" ]]; then
    return 0
  fi

  shopt -s nullglob
  local cache_dbs=("${LEGACY_CACHE_MINI_DIR}"/*.db)
  shopt -u nullglob
  if (( ${#cache_dbs[@]} == 0 )); then
    return 0
  fi

  echo "[bootstrap] Normalizing mini DB layout: ${LEGACY_CACHE_MINI_DIR} -> ${MINI_DIR}"
  mkdir -p "${MINI_DIR}"
  local db=""
  for db in "${cache_dbs[@]}"; do
    mv -f "${db}" "${MINI_DIR}/"
  done

  # Best-effort cleanup of now-empty legacy cache path.
  rmdir "${LEGACY_CACHE_MINI_DIR}" 2>/dev/null || true
  rmdir "${DATA_ROOT}/data/cache" 2>/dev/null || true
  rmdir "${DATA_ROOT}/data" 2>/dev/null || true
}

bool() {
  case "${1:-false}" in
    true|TRUE|1|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

mini_dbs_are_valid() {
  local dir="$1"
  local db=""

  if ! command -v sqlite3 >/dev/null 2>&1; then
    echo "[bootstrap] sqlite3 not found; skipping mini DB integrity checks."
    return 0
  fi

  shopt -s nullglob
  for db in "${dir}"/*.db; do
    local qc_out
    qc_out="$(sqlite3 "${db}" 'PRAGMA quick_check;' 2>/dev/null | tr -d '\r' | head -n 1 || true)"
    if [[ "${qc_out}" != "ok" ]]; then
      echo "[bootstrap] Corrupted mini DB detected: ${db}" >&2
      return 1
    fi
  done
  shopt -u nullglob
  return 0
}

print_plan() {
  cat <<PLAN
[nuPlan bootstrap plan]
  profile_file: ${PROFILE_FILE}
  data_root:    ${DATA_ROOT}
  maps_root:    ${MAPS_ROOT}
  exp_root:     ${EXP_ROOT}

  Download components:
    HD semantic maps:       ${DOWNLOAD_MAPS:-false}
    mini SQLite log DBs:    ${DOWNLOAD_MINI_DB:-false}
    mini camera blobs:      ${DOWNLOAD_MINI_CAMERA:-false}
    mini LiDAR blobs:       ${DOWNLOAD_MINI_LIDAR:-false}

  Planner-essential content:
    - Ego pose/state:        inside SQLite log DBs
    - Agent tracks:          inside SQLite log DBs
    - Traffic lights:        inside SQLite log DBs
    - Route/mission metadata: inside SQLite log DBs
    - HD semantic map:       separate map archive
    - Simulator/metrics:     installed as nuPlan devkit code, not dataset files
    - nuBoard:               installed as nuPlan devkit code, not dataset files

  Active mini DB limit:      ${MAX_ACTIVE_MINI_DBS:-all}
  Trim after extract:        ${TRIM_MINI_DBS_AFTER_EXTRACT:-false}
  Trim mode:                 ${TRIM_MODE:-none}
  Keep archives:             ${KEEP_ARCHIVES:-true}

Notes:
  - The official public mini DB is distributed as one zip; this script cannot
    partially download individual DBs from that zip. It can only skip the whole
    DB archive, or optionally reduce the active/unpacked set after extraction.
  - Full train/val/test archives usually require official login/terms acceptance.
PLAN
}

fetch() {
  local url="$1"
  local out="$2"
  if [[ -f "${out}" && "${FORCE}" != "true" ]]; then
    echo "[bootstrap] Archive exists: ${out}"
    return 0
  fi
  echo "[bootstrap] Downloading ${url}"
  wget --continue --tries=5 --timeout=30 -O "${out}" "${url}"
}

unzip_once() {
  local zip_path="$1"
  local target="$2"
  local marker_glob="$3"
  if compgen -G "${marker_glob}" >/dev/null && [[ "${FORCE}" != "true" ]]; then
    echo "[bootstrap] Already unpacked matching: ${marker_glob}"
    return 0
  fi

  local avail_bytes=0
  local uncompressed_bytes=0
  local required_bytes=0
  local safety_bytes=$((1024 * 1024 * 1024))
  local zipinfo_meta=""
  avail_bytes="$(df -PB1 "${target}" | awk 'NR==2 {print $4}')"

  if zipinfo_meta="$(zipinfo -t "${zip_path}" 2>/dev/null)"; then
    uncompressed_bytes="$(echo "${zipinfo_meta}" | sed -nE 's/.* ([0-9][0-9,]*) bytes uncompressed.*/\1/p' | tr -d ',' | tail -n 1)"
  fi
  if [[ ! "${uncompressed_bytes}" =~ ^[0-9]+$ ]]; then
    # Fallback when archive metadata parsing is unavailable.
    local zip_size
    zip_size="$(stat -c %s "${zip_path}")"
    uncompressed_bytes=$((zip_size + (zip_size / 3)))
  fi

  required_bytes=$((uncompressed_bytes + safety_bytes))
  if (( avail_bytes < required_bytes )); then
    echo "[bootstrap] ERROR: insufficient free disk space before unpacking $(basename "${zip_path}")." >&2
    echo "[bootstrap] Available bytes: ${avail_bytes}" >&2
    echo "[bootstrap] Required bytes (estimate): ${required_bytes} (uncompressed archive + 1 GiB safety)." >&2
    echo "[bootstrap] Free space and rerun bootstrap." >&2
    exit 50
  fi

  echo "[bootstrap] Unpacking ${zip_path} -> ${target}"
  unzip -q -o "${zip_path}" -d "${target}"
}

write_active_manifest() {
  local max_count="${MAX_ACTIVE_MINI_DBS:-all}"
  : > "${ACTIVE_MANIFEST}"
  if [[ "${max_count}" == "all" ]]; then
    find "${MINI_DIR}" -maxdepth 1 -type f -name '*.db' | sort > "${ACTIVE_MANIFEST}" || true
  elif [[ "${max_count}" =~ ^[0-9]+$ ]]; then
    find "${MINI_DIR}" -maxdepth 1 -type f -name '*.db' | sort | head -n "${max_count}" > "${ACTIVE_MANIFEST}" || true
  else
    echo "MAX_ACTIVE_MINI_DBS must be 'all' or an integer; got '${max_count}'" >&2
    exit 2
  fi
  echo "[bootstrap] Active mini DB manifest: ${ACTIVE_MANIFEST}"
  echo "[bootstrap] Active mini DB count: $(wc -l < "${ACTIVE_MANIFEST}" | tr -d ' ')"
}

trim_mini_dbs_if_requested() {
  if ! bool "${TRIM_MINI_DBS_AFTER_EXTRACT:-false}"; then
    return 0
  fi
  local max_count="${MAX_ACTIVE_MINI_DBS:-all}"
  local trim_mode="${TRIM_MODE:-none}"
  if [[ "${max_count}" == "all" || "${trim_mode}" == "none" ]]; then
    return 0
  fi
  if ! [[ "${max_count}" =~ ^[0-9]+$ ]]; then
    echo "Cannot trim: MAX_ACTIVE_MINI_DBS must be an integer." >&2
    exit 2
  fi

  mapfile -t keep < <(find "${MINI_DIR}" -maxdepth 1 -type f -name '*.db' | sort | head -n "${max_count}")
  mapfile -t all < <(find "${MINI_DIR}" -maxdepth 1 -type f -name '*.db' | sort)
  mkdir -p "${MINI_DISABLED_DIR}"

  for db in "${all[@]}"; do
    local keep_it="false"
    for k in "${keep[@]}"; do
      if [[ "${db}" == "${k}" ]]; then keep_it="true"; break; fi
    done
    if [[ "${keep_it}" == "false" ]]; then
      case "${trim_mode}" in
        disable)
          echo "[bootstrap] Disabling inactive DB: $(basename "${db}")"
          mv "${db}" "${MINI_DISABLED_DIR}/"
          ;;
        delete)
          echo "[bootstrap] Deleting inactive DB: $(basename "${db}")"
          rm -f "${db}"
          ;;
        none)
          ;;
        *)
          echo "Unknown TRIM_MODE='${trim_mode}'. Use none, disable, or delete." >&2
          exit 2
          ;;
      esac
    fi
  done
}

print_plan
if [[ "${PRINT_ONLY}" == "true" ]]; then
  exit 0
fi

if bool "${DOWNLOAD_MAPS:-false}"; then
  if [[ -f "${MAPS_ROOT}/nuplan-maps-v1.0.json" && "${FORCE}" != "true" ]]; then
    echo "[bootstrap] Already unpacked matching: ${MAPS_ROOT}/nuplan-maps-v1.0.json"
  else
    fetch "${NUPLAN_MAPS_URL}" "${ARCHIVE_DIR}/nuplan-maps-v1.0.zip"
    unzip_once "${ARCHIVE_DIR}/nuplan-maps-v1.0.zip" "${DATA_ROOT}" "${MAPS_ROOT}/nuplan-maps-v1.0.json"
  fi
else
  echo "[bootstrap] Skipping maps by profile."
fi

if bool "${DOWNLOAD_MINI_DB:-false}"; then
  normalize_mini_dir
  if compgen -G "${MINI_DIR}/*.db" >/dev/null && [[ "${FORCE}" != "true" ]]; then
    if mini_dbs_are_valid "${MINI_DIR}"; then
      echo "[bootstrap] Already unpacked matching: ${MINI_DIR}/*.db"
    else
      echo "[bootstrap] Re-unpacking mini DB archive because corrupted DB files were detected."
      fetch "${NUPLAN_MINI_DB_URL}" "${ARCHIVE_DIR}/nuplan-v1.1_mini.zip"
      unzip_once "${ARCHIVE_DIR}/nuplan-v1.1_mini.zip" "${DATA_ROOT}" "${MINI_DIR}/*.db"
      normalize_mini_dir
    fi
  else
    fetch "${NUPLAN_MINI_DB_URL}" "${ARCHIVE_DIR}/nuplan-v1.1_mini.zip"
    unzip_once "${ARCHIVE_DIR}/nuplan-v1.1_mini.zip" "${DATA_ROOT}" "${MINI_DIR}/*.db"
    normalize_mini_dir
  fi
  echo "[bootstrap] Using mini DB dir: ${MINI_DIR}"
  trim_mini_dbs_if_requested
  write_active_manifest
else
  echo "[bootstrap] Skipping mini SQLite log DBs by profile."
fi

if bool "${DOWNLOAD_MINI_CAMERA:-false}"; then
  if compgen -G "${SENSOR_DIR}/*/CAM_*" >/dev/null && [[ "${FORCE}" != "true" ]]; then
    echo "[bootstrap] Already unpacked matching: ${SENSOR_DIR}/*/CAM_*"
  else
    fetch "${NUPLAN_MINI_CAMERA_0_URL}" "${ARCHIVE_DIR}/nuplan-v1.1_mini_camera_0.zip"
    unzip_once "${ARCHIVE_DIR}/nuplan-v1.1_mini_camera_0.zip" "${DATA_ROOT}" "${SENSOR_DIR}/*/CAM_*"
  fi
else
  echo "[bootstrap] Skipping mini camera blobs by profile."
fi

if bool "${DOWNLOAD_MINI_LIDAR:-false}"; then
  if compgen -G "${SENSOR_DIR}/*/MergedPointCloud/*" >/dev/null && [[ "${FORCE}" != "true" ]]; then
    echo "[bootstrap] Already unpacked matching: ${SENSOR_DIR}/*/MergedPointCloud/*"
  else
    fetch "${NUPLAN_MINI_LIDAR_0_URL}" "${ARCHIVE_DIR}/nuplan-v1.1_mini_lidar_0.zip"
    unzip_once "${ARCHIVE_DIR}/nuplan-v1.1_mini_lidar_0.zip" "${DATA_ROOT}" "${SENSOR_DIR}/*/MergedPointCloud/*"
  fi
else
  echo "[bootstrap] Skipping mini LiDAR blobs by profile."
fi

if ! bool "${KEEP_ARCHIVES:-true}"; then
  echo "[bootstrap] Removing archives in ${ARCHIVE_DIR}"
  rm -f "${ARCHIVE_DIR}"/*.zip
fi

normalize_mini_dir
echo
echo "[bootstrap] Done."
echo "[bootstrap] mini DBs present: $(find "${MINI_DIR}" -maxdepth 1 -type f -name '*.db' | wc -l | tr -d ' ')"
echo "[bootstrap] maps json present: $([[ -f "${MAPS_ROOT}/nuplan-maps-v1.0.json" ]] && echo yes || echo no)"
echo "[bootstrap] sensor blob dir present: $([[ -d "${SENSOR_DIR}" ]] && echo yes || echo no)"