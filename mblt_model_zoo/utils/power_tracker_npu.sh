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
    printf '{"ok":true,"npu_power_w":%s,"total_power_w":%s,"timestamp":%s}\n' \
        "$npu_power_w" "$total_power_w" "$timestamp"
else
    printf '%s %s\n' "$npu_power_w" "$total_power_w"
fi
