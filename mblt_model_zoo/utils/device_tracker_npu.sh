#!/usr/bin/env bash

set -u

JSON_MODE=false
SAMPLE_ONCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample-once)
            SAMPLE_ONCE=true
            shift
            ;;
        --json)
            JSON_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --sample-once --json"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if [[ "$SAMPLE_ONCE" != "true" ]]; then
    echo "Error: only --sample-once mode is supported." >&2
    exit 2
fi

if ! command -v mobilint-cli >/dev/null 2>&1; then
    if [[ "$JSON_MODE" == "true" ]]; then
        printf '{"ok":false,"error":"mobilint-cli not found","timestamp":%s}\n' "$(date +%s)"
    else
        echo "mobilint-cli not found" >&2
    fi
    exit 1
fi

status_output="$(mobilint-cli status 2>/dev/null)"
if [[ -z "$status_output" ]]; then
    if [[ "$JSON_MODE" == "true" ]]; then
        printf '{"ok":false,"error":"empty status output","timestamp":%s}\n' "$(date +%s)"
    else
        echo "empty status output" >&2
    fi
    exit 1
fi

# Remove ANSI escape sequences if CLI outputs colored table text.
status_output="$(printf '%s' "$status_output" | sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g')"

# Expected power line shape: |   X.XXW  Y.YYW |
power_line="$(
    echo "$status_output" \
    | grep -E '\|\s*[0-9]+(\.[0-9]+)?W\s+[0-9]+(\.[0-9]+)?W\s+\|' \
    | head -n 1
)"
if [[ -z "$power_line" ]]; then
    if [[ "$JSON_MODE" == "true" ]]; then
        printf '{"ok":false,"error":"power line not found","timestamp":%s}\n' "$(date +%s)"
    else
        echo "power line not found" >&2
    fi
    exit 1
fi

npu_power_w="$(
    echo "$power_line" \
    | sed -E 's/^.*\|\s*([0-9]+(\.[0-9]+)?)W\s+([0-9]+(\.[0-9]+)?)W\s+\|.*$/\1/'
)"
total_power_w="$(
    echo "$power_line" \
    | sed -E 's/^.*\|\s*([0-9]+(\.[0-9]+)?)W\s+([0-9]+(\.[0-9]+)?)W\s+\|.*$/\3/'
)"
npu_util_pct="$(
    echo "$status_output" \
    | awk '/\|[[:space:]]*[0-9]+(\.[0-9]+)?W[[:space:]]+[0-9]+(\.[0-9]+)?W[[:space:]]+\|/ {getline; print; exit}' \
    | sed -nE 's/^.*\|\s*([0-9]+(\.[0-9]+)?)%\s+\|.*$/\1/p'
)"
memory_line="$(
    echo "$status_output" \
    | grep -E '\|\s*[0-9]+(\.[0-9]+)?W\s+[0-9]+(\.[0-9]+)?W\s+\|.*\|\s*[0-9]+(\.[0-9]+)?MB\s*/\s*[0-9]+(\.[0-9]+)?MB\s+\|' \
    | head -n 1
)"
npu_mem_used_mb="$(
    echo "$memory_line" \
    | sed -nE 's/^.*\|\s*([0-9]+(\.[0-9]+)?)MB\s*\/\s*([0-9]+(\.[0-9]+)?)MB\s+\|.*$/\1/p'
)"
npu_mem_total_mb="$(
    echo "$memory_line" \
    | sed -nE 's/^.*\|\s*([0-9]+(\.[0-9]+)?)MB\s*\/\s*([0-9]+(\.[0-9]+)?)MB\s+\|.*$/\3/p'
)"
npu_mem_used_pct=""
if [[ -n "$npu_mem_used_mb" && -n "$npu_mem_total_mb" ]]; then
    npu_mem_used_pct="$(
        awk -v used="$npu_mem_used_mb" -v total="$npu_mem_total_mb" 'BEGIN { if (total > 0) printf "%.6f", (used/total)*100.0; }'
    )"
fi
timestamp="$(date +%s)"

if [[ -z "$npu_power_w" || -z "$total_power_w" ]]; then
    if [[ "$JSON_MODE" == "true" ]]; then
        printf '{"ok":false,"error":"failed to parse power values","timestamp":%s}\n' "$timestamp"
    else
        echo "failed to parse power values" >&2
    fi
    exit 1
fi

if [[ "$JSON_MODE" == "true" ]]; then
    json_fields='"ok":true'
    json_fields="$json_fields,\"npu_power_w\":$npu_power_w"
    json_fields="$json_fields,\"total_power_w\":$total_power_w"
    if [[ -n "$npu_util_pct" ]]; then
        json_fields="$json_fields,\"npu_util_pct\":$npu_util_pct"
    fi
    if [[ -n "$npu_mem_used_mb" ]]; then
        json_fields="$json_fields,\"npu_mem_used_mb\":$npu_mem_used_mb"
    fi
    if [[ -n "$npu_mem_total_mb" ]]; then
        json_fields="$json_fields,\"npu_mem_total_mb\":$npu_mem_total_mb"
    fi
    if [[ -n "$npu_mem_used_pct" ]]; then
        json_fields="$json_fields,\"npu_mem_used_pct\":$npu_mem_used_pct"
    fi
    json_fields="$json_fields,\"timestamp\":$timestamp"
    printf '{%s}\n' "$json_fields"
else
    printf '%s %s %s %s %s %s\n' \
        "$npu_power_w" \
        "$total_power_w" \
        "${npu_util_pct:-}" \
        "${npu_mem_used_mb:-}" \
        "${npu_mem_total_mb:-}" \
        "${npu_mem_used_pct:-}"
fi
