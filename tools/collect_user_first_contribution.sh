#!/usr/bin/env bash

set -euo pipefail

# ==============================================================================
# Default configuration
# ==============================================================================

DEFAULT_REPO="vllm-project/vllm-ascend"
DEFAULT_CONTRIBUTORS_FILE="docs/source/community/contributors.md"
LINK_CHECK_JOBS=4

REPO="${DEFAULT_REPO}"
CONTRIBUTORS_FILE="${DEFAULT_CONTRIBUTORS_FILE}"
FORCE_FULL=false
ENABLE_LINK_CHECK=false
SORT_ONLY=false

SCRIPT_DIR=""
PROJECT_ROOT=""
CURRENT_HEAD=""
CURRENT_HEAD_SHORT=""
LAST_COMMIT=""
INCREMENTAL=false
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

TEMP_FILES=()

# ==============================================================================
# Common helpers
# ==============================================================================

usage() {
  echo "This script collects contributors' first contributions and updates the contributors.md file."
  echo "Supports incremental updates by tracking the last commit hash."
  echo ""
  echo "Please set the environment variable GITHUB_TOKEN with repo read permission unless using --sort-only."
  echo "Refer to https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api?apiVersion=2022-11-28"
  echo ""
  echo "Usage: $0 [options]"
  echo "       $0 --full  # Force full refresh (ignore last commit hash)"
  echo "       $0 --help"
  echo ""
  echo "Options:"
  echo "  --full             Force full refresh, recalculate all contributors"
  echo "  --link-check       Check GitHub profile and commit links for final contributors"
  echo "  --sort-only        Refresh contributor table numbers only, without GitHub API access"
  echo "  --repo=OWNER/REPO  Specify GitHub repository (default: ${DEFAULT_REPO})"
  echo "  --file=PATH        Specify contributors.md path (default: ${DEFAULT_CONTRIBUTORS_FILE})"
  echo ""
  echo "Examples:"
  echo "  $0                 # Incremental update from last commit"
  echo "  $0 --full          # Full refresh"
  echo "  $0 --link-check    # Incremental update with link checking"
  echo "  $0 --sort-only     # Refresh contributor table numbers only"
}

die() {
  echo "Error: $*" >&2
  exit 1
}

make_temp_file() {
  local var_name="$1"
  local temp_file

  temp_file=$(mktemp)
  TEMP_FILES+=("$temp_file")

  printf -v "$var_name" "%s" "$temp_file"
}

cleanup_temp_files() {
  if [ "${#TEMP_FILES[@]}" -gt 0 ]; then
    rm -f "${TEMP_FILES[@]}"
  fi
}

# ==============================================================================
# Basic utility functions
# ==============================================================================

get_last_commit_hash() {
  local file="$1"

  grep -o '<!-- last_commit: [a-f0-9]* -->' "$file" 2>/dev/null \
    | sed 's/<!-- last_commit: \([a-f0-9]*\) -->/\1/' \
    || echo ""
}

get_current_contributor_count() {
  local file="$1"

  grep -o '| [0-9]* |' "$file" 2>/dev/null \
    | head -1 \
    | grep -o '[0-9]*' \
    || echo "0"
}

format_commit_date() {
  local date="$1"
  local short_date

  short_date="${date%%T*}"
  echo "${short_date//-//}"
}

format_contributor_row() {
  local number="$1"
  local login="$2"
  local date="$3"
  local sha="$4"
  local short_sha

  short_sha="${sha:0:7}"
  printf "| %s | [@%s](https://github.com/%s) | %s | [%s](https://github.com/%s/commit/%s) |" \
    "$number" "$login" "$login" "$date" "$short_sha" "$REPO" "$sha"
}

extract_existing_logins() {
  local file="$1"

  sed -n 's#^|[[:space:]]*[0-9][0-9]*[[:space:]]*|[^|]*(https://github.com/\([^)]*\))[[:space:]]*|[[:space:]]*[0-9].*#\1#p' \
    "$file" 2>/dev/null \
    | sort -u \
    || true
}

extract_login_from_noreply_email() {
  local email="$1"

  if [[ "$email" == *@users.noreply.github.com ]]; then
    local local_part="${email%@users.noreply.github.com}"

    if [[ "$local_part" == *+* ]]; then
      echo "${local_part#*+}"
    else
      echo "$local_part"
    fi
  else
    echo ""
  fi
}

get_github_login() {
  local sha="$1"
  local email="$2"
  local api_url
  local resp
  local login

  api_url="https://api.github.com/repos/${REPO}/commits/${sha}"

  resp=$(curl -g -s \
    -H "Authorization: token ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github.v3+json" \
    "$api_url" || true)

  login=$(echo "$resp" | jq -r '.author.login // empty' 2>/dev/null || echo "")

  if [ -z "$login" ]; then
    login=$(extract_login_from_noreply_email "$email")
  fi

  echo "$login"
}

get_github_http_code() {
  local api_url="$1"
  local http_code

  http_code=$(curl -g -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: token ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github.v3+json" \
    "$api_url" || true)

  if [ -z "$http_code" ]; then
    echo "000"
  else
    echo "$http_code"
  fi
}

collect_contributors_by_login() {
  local all_history="$1"
  local contributors_data="$2"
  local login_commits="$3"

  local SORTED_HISTORY
  local EMAIL_FIRST_COMMITS
  local EMAIL_LOGINS

  make_temp_file SORTED_HISTORY
  make_temp_file EMAIL_FIRST_COMMITS
  make_temp_file EMAIL_LOGINS

  # 1. Sort all commits by commit time so email and login groups are stable.
  sort -t'|' -k1,1n "$all_history" > "$SORTED_HISTORY"

  # 2. Collapse commits by author email and keep each email's earliest commit.
  #    The git log contains author name too, but downstream only needs email,
  #    commit SHA, and date for table generation.
  awk -F'|' '!seen[$3]++ {print $3 "|" $2 "|" $4}' \
    "$SORTED_HISTORY" > "$EMAIL_FIRST_COMMITS"

  # 3. Resolve each author email to a GitHub login exactly once.
  local TOTAL
  local CURRENT=0

  TOTAL=$(wc -l < "$EMAIL_FIRST_COMMITS" | tr -d ' ')
  echo "Processing ${TOTAL} author emails..."

  while IFS='|' read -r email sha _date; do
    CURRENT=$((CURRENT + 1))
    printf "\rProcessing: %d/%d" "$CURRENT" "$TOTAL"

    local login
    login=$(get_github_login "$sha" "$email")

    if [ -n "$login" ]; then
      echo "${email}|${login}" >> "$EMAIL_LOGINS"
    fi
  done < "$EMAIL_FIRST_COMMITS"

  echo ""
  echo ""

  # 4. Merge all commits from all emails under their resolved GitHub login.
  #    Format: login|sha|date|email|timestamp
  awk -F'|' '
    NR == FNR {
      login_by_email[$1] = $2
      next
    }
    {
      login = login_by_email[$3]
      if (login != "") {
        print login "|" $2 "|" $4 "|" $3 "|" $1
      }
    }' "$EMAIL_LOGINS" "$SORTED_HISTORY" \
    | sort -t'|' -k1,1 -k5,5n > "$login_commits"

  # 5. Pick the earliest commit for each GitHub login after email merging.
  #    Format: login|sha|short_sha|formatted_date|timestamp
  awk -F'|' '
    !seen[$1]++ {
      formatted_date = $3
      sub(/T.*/, "", formatted_date)
      gsub(/-/, "/", formatted_date)
      print $1 "|" $2 "|" substr($2, 1, 7) "|" formatted_date "|" $5
    }' "$login_commits" > "$contributors_data"
}

generate_numbered_contributors() {
  local contributors_data="$1"
  local numbered_contributors="$2"
  local start_number="$3"
  local mode="$4"

  if [ "$mode" = "full" ]; then
    sort -t'|' -k5,5nr "$contributors_data" \
      | awk -F'|' -v total="$start_number" '
        BEGIN { nr = total }
        {
          print nr "|" $0
          nr--
        }' > "$numbered_contributors"
  else
    local contributor_count
    contributor_count=$(wc -l < "$contributors_data" | tr -d ' ')

    sort -t'|' -k5,5nr "$contributors_data" \
      | awk -F'|' -v total="$((start_number + contributor_count))" '
        BEGIN { nr = total }
        {
          print nr "|" $0
          nr--
        }' > "$numbered_contributors"
  fi
}

write_contributor_rows() {
  local numbered_contributors="$1"
  local output_file="$2"

  if [ "$output_file" = "-" ]; then
    awk -F'|' -v repo="$REPO" '
      {
        number = $1
        login = $2
        sha = $3
        short_sha = $4
        date = $5

        printf "| %d | [@%s](https://github.com/%s) | %s | [%s](https://github.com/%s/commit/%s) |\n", number, login, login, date, short_sha, repo, sha
      }' "$numbered_contributors"
    return
  fi

  awk -F'|' -v repo="$REPO" '
    {
      number = $1
      login = $2
      sha = $3
      short_sha = $4
      date = $5

      printf "| %d | [@%s](https://github.com/%s) | %s | [%s](https://github.com/%s/commit/%s) |\n", number, login, login, date, short_sha, repo, sha
    }' "$numbered_contributors" > "$output_file"
}

find_valid_commit_replacement() {
  local number="$1"
  local login="$2"
  local selected_sha="$3"
  local login_commits="$4"
  local found_selected=false

  while IFS='|' read -r commit_login commit_sha commit_date _; do
    if [ "$commit_login" != "$login" ]; then
      continue
    fi

    if [ "$found_selected" != true ]; then
      if [ "$commit_sha" = "$selected_sha" ]; then
        found_selected=true
      fi
      continue
    fi

    local commit_code
    commit_code=$(get_github_http_code "https://api.github.com/repos/${REPO}/commits/${commit_sha}")

    if [ "$commit_code" = "200" ]; then
      local replacement_date
      replacement_date=$(format_commit_date "$commit_date")
      format_contributor_row "$number" "$login" "$replacement_date" "$commit_sha"
      return
    fi
  done < "$login_commits"

  echo ""
}

run_link_check_one() {
  local number="$1"
  local login="$2"
  local sha="$3"
  local date="$4"
  local login_commits="$2"

  login_commits="$5"

  local row
  local profile_code
  local commit_code

  row=$(format_contributor_row "$number" "$login" "$date" "$sha")
  profile_code=$(get_github_http_code "https://api.github.com/users/${login}")

  if [ "$profile_code" != "200" ]; then
    echo "${row} ------ Invalid profile(${profile_code})"
    return
  fi

  commit_code=$(get_github_http_code "https://api.github.com/repos/${REPO}/commits/${sha}")

  if [ "$commit_code" != "200" ]; then
    local replacement_row

    echo "${row} ------ Invalid commit(${commit_code})"
    replacement_row=$(find_valid_commit_replacement "$number" "$login" "$sha" "$login_commits")

    if [ -n "$replacement_row" ]; then
      echo "==>"
      echo "$replacement_row"
    fi
  fi
}

run_link_check() {
  local numbered_contributors="$1"
  local login_commits="$2"

  if [ "$ENABLE_LINK_CHECK" != true ]; then
    return
  fi

  echo ""
  echo "Checking GitHub profile and commit links with ${LINK_CHECK_JOBS} jobs..."

  local checked=0
  local issues=0
  local batch_size=0
  local pids=()
  local output_files=()

  while IFS='|' read -r number login sha _short_sha date _timestamp; do
    if [ -z "$login" ]; then
      continue
    fi

    checked=$((checked + 1))

    local CHECK_OUTPUT
    make_temp_file CHECK_OUTPUT
    output_files+=("$CHECK_OUTPUT")

    run_link_check_one "$number" "$login" "$sha" "$date" "$login_commits" > "$CHECK_OUTPUT" &
    pids+=("$!")
    batch_size=$((batch_size + 1))

    if [ "$batch_size" -ge "$LINK_CHECK_JOBS" ]; then
      local pid
      for pid in "${pids[@]}"; do
        wait "$pid" || true
      done
      pids=()
      batch_size=0
    fi
  done < "$numbered_contributors"

  local pid
  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done

  local output_file
  for output_file in "${output_files[@]}"; do
    if [ -s "$output_file" ]; then
      cat "$output_file"
      issues=$((issues + 1))
    fi
  done

  echo "Link check completed: ${checked} checked, ${issues} issue(s) found."
}

# ==============================================================================
# Step 1: Parse arguments
# ==============================================================================

parse_args() {
  for arg in "$@"; do
    case "$arg" in
      --help)
        usage
        exit 0
        ;;
      --full)
        FORCE_FULL=true
        ;;
      --link-check)
        ENABLE_LINK_CHECK=true
        ;;
      --sort-only)
        SORT_ONLY=true
        ;;
      --repo=*)
        REPO="${arg#*=}"
        ;;
      --file=*)
        CONTRIBUTORS_FILE="${arg#*=}"
        ;;
      *)
        echo "Unknown argument: $arg"
        usage
        exit 1
        ;;
    esac
  done

  if [ "$SORT_ONLY" = true ] && [ "$FORCE_FULL" = true ]; then
    die "--sort-only cannot be used with --full."
  fi

  if [ "$SORT_ONLY" = true ] && [ "$ENABLE_LINK_CHECK" = true ]; then
    die "--sort-only cannot be used with --link-check."
  fi
}

# ==============================================================================
# Step 2: Initialize script context
# ==============================================================================

init_context() {
  # 1. Check GitHub token.
  GITHUB_TOKEN="${GITHUB_TOKEN:-}"

  if [ "$SORT_ONLY" != true ] && [ -z "$GITHUB_TOKEN" ]; then
    die "Please set the environment variable GITHUB_TOKEN with repo read permission."
  fi

  # 2. Locate project root.
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

  # 3. Resolve contributors.md path.
  if [[ "$CONTRIBUTORS_FILE" != /* ]]; then
    CONTRIBUTORS_FILE="${PROJECT_ROOT}/${CONTRIBUTORS_FILE}"
  fi

  if [ ! -f "$CONTRIBUTORS_FILE" ]; then
    die "Contributors file not found: ${CONTRIBUTORS_FILE}"
  fi

  # 4. Enter project root for git operations.
  cd "$PROJECT_ROOT"

  # 5. Get current HEAD.
  CURRENT_HEAD=$(git rev-parse HEAD)
  CURRENT_HEAD_SHORT="${CURRENT_HEAD:0:7}"

  echo "Repository: ${REPO}"
  echo "Contributors file: ${CONTRIBUTORS_FILE}"
  echo "Current HEAD: ${CURRENT_HEAD_SHORT}"
  echo ""
}

# ==============================================================================
# Step 3: Decide update mode
# ==============================================================================

determine_update_mode() {
  LAST_COMMIT=""
  INCREMENTAL=false

  # If --full is specified, skip incremental check.
  if [ "$FORCE_FULL" = true ]; then
    return
  fi

  # 1. Read last recorded commit from contributors.md.
  LAST_COMMIT=$(get_last_commit_hash "$CONTRIBUTORS_FILE")

  # 2. If last_commit exists and differs from current HEAD,
  #    check whether incremental update is safe.
  if [ -n "$LAST_COMMIT" ] && [ "$LAST_COMMIT" != "$CURRENT_HEAD" ]; then
    if git merge-base --is-ancestor "$LAST_COMMIT" "$CURRENT_HEAD" 2>/dev/null; then
      INCREMENTAL=true
      echo "Incremental update from commit: ${LAST_COMMIT:0:7}"
    else
      echo "Warning: Last commit ${LAST_COMMIT:0:7} is not an ancestor of current HEAD, doing full refresh."
    fi

  # 3. If last_commit equals current HEAD, no update is needed.
  elif [ "$LAST_COMMIT" = "$CURRENT_HEAD" ]; then
    echo "Already up to date (HEAD matches last recorded commit)."
    echo "Use --full to force a full refresh."
    exit 0
  fi
}

# ==============================================================================
# Step 4A: Incremental update
# ==============================================================================

run_incremental_update() {
  echo ""
  echo "Fetching new commits..."

  local ALL_HISTORY
  local NEW_SHAS
  local EXISTING_LOGINS
  local CONTRIBUTORS_DATA
  local LOGIN_COMMITS
  local NEW_CONTRIBUTORS
  local NUMBERED_CONTRIBUTORS
  local NEW_ROWS
  local TEMP_FILE

  make_temp_file ALL_HISTORY
  make_temp_file NEW_SHAS
  make_temp_file EXISTING_LOGINS
  make_temp_file CONTRIBUTORS_DATA
  make_temp_file LOGIN_COMMITS
  make_temp_file NEW_CONTRIBUTORS
  make_temp_file NUMBERED_CONTRIBUTORS
  make_temp_file NEW_ROWS
  make_temp_file TEMP_FILE

  # ---------------------------------------------------------------------------
  # 1. Collect git data.
  # ---------------------------------------------------------------------------

  # Full git history is needed to merge emails into GitHub logins correctly.
  git log --pretty=format:'%ct|%H|%aE|%cI|%aN' \
    --reverse --all > "$ALL_HISTORY"

  # Extract all new commit SHAs in this incremental range.
  git rev-list "${LAST_COMMIT}..${CURRENT_HEAD}" > "$NEW_SHAS"

  # Extract existing GitHub logins from contributors.md for deduplication.
  extract_existing_logins "$CONTRIBUTORS_FILE" > "$EXISTING_LOGINS"

  # ---------------------------------------------------------------------------
  # 2. Merge commits by email, then by GitHub login.
  # ---------------------------------------------------------------------------

  collect_contributors_by_login "$ALL_HISTORY" "$CONTRIBUTORS_DATA" "$LOGIN_COMMITS"

  # ---------------------------------------------------------------------------
  # 3. Find new contributors.
  # ---------------------------------------------------------------------------

  local skipped=0

  while IFS='|' read -r login sha short_sha formatted_date timestamp; do
    # Only keep logins whose final first contribution is in the new range.
    if ! grep -Fxq "$sha" "$NEW_SHAS"; then
      continue
    fi

    # Skip contributors that already exist in contributors.md.
    if grep -Fxq "$login" "$EXISTING_LOGINS"; then
      echo "Skipping duplicate contributor: $login"
      ((skipped++)) || true
      continue
    fi

    echo "${login}|${sha}|${short_sha}|${formatted_date}|${timestamp}" >> "$NEW_CONTRIBUTORS"
  done < "$CONTRIBUTORS_DATA"

  # ---------------------------------------------------------------------------
  # 4. Check whether there are new contributors.
  # ---------------------------------------------------------------------------

  local NEW_COUNT
  NEW_COUNT=$(wc -l < "$NEW_CONTRIBUTORS" | tr -d ' ')

  echo "Found ${NEW_COUNT} new contributors"

  if [ "$skipped" -gt 0 ]; then
    echo "Skipped ${skipped} duplicate contributors"
  fi

  if [ "$NEW_COUNT" -eq 0 ]; then
    echo "No new contributors found."
    exit 0
  fi

  # ---------------------------------------------------------------------------
  # 5. Generate new contributor rows.
  # ---------------------------------------------------------------------------

  local CURRENT_COUNT
  local NEW_TOTAL

  CURRENT_COUNT=$(get_current_contributor_count "$CONTRIBUTORS_FILE")
  NEW_TOTAL=$((CURRENT_COUNT + NEW_COUNT))

  echo "Current contributor count: ${CURRENT_COUNT}"

  generate_numbered_contributors "$NEW_CONTRIBUTORS" "$NUMBERED_CONTRIBUTORS" "$CURRENT_COUNT" "incremental"
  write_contributor_rows "$NUMBERED_CONTRIBUTORS" "$NEW_ROWS"
  run_link_check "$NUMBERED_CONTRIBUTORS" "$LOGIN_COMMITS"

  # ---------------------------------------------------------------------------
  # 6. Rewrite contributors.md.
  # ---------------------------------------------------------------------------

  local CURRENT_DATE
  local WROTE_HEADER=false

  CURRENT_DATE=$(date +%Y-%m-%d)

  while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" == "<!-- last_commit:"* ]]; then
      # Remove old last_commit.
      continue

    elif [[ "$line" == "Updated on "* ]]; then
      # Remove old update date.
      continue

    elif [[ "$line" == "Every release of vLLM Ascend"* ]]; then
      # Remove old description.
      continue

    elif [[ "$line" == "| Number | Contributor | Date | Commit ID |" ]]; then
      # Insert new metadata before table header.
      {
        echo "<!-- last_commit: ${CURRENT_HEAD} -->"
        echo ""
        echo "Every release of vLLM Ascend would not have been possible without the following contributors:"
        echo ""
        echo "Updated on ${CURRENT_DATE}:"
        echo ""
        echo "$line"
      } >> "$TEMP_FILE"
      WROTE_HEADER=true

    elif [[ "$WROTE_HEADER" == true && "$line" == "|:"* ]]; then
      # Insert new rows right after table separator.
      echo "$line" >> "$TEMP_FILE"
      cat "$NEW_ROWS" >> "$TEMP_FILE"
      WROTE_HEADER=false

    else
      # Keep existing contributor numbers unchanged. New rows receive the
      # highest numbers, so existing first-contribution order remains stable.
      echo "$line" >> "$TEMP_FILE"
    fi
  done < "$CONTRIBUTORS_FILE"

  mv "$TEMP_FILE" "$CONTRIBUTORS_FILE"

  echo ""
  echo "Done! Added ${NEW_COUNT} new contributors. Total: ${NEW_TOTAL}"
}

# ==============================================================================
# Step 4B: Full refresh
# ==============================================================================

run_full_refresh() {
  echo "Performing full refresh..."
  echo ""

  local ALLCOMMITS
  local CONTRIBUTORS_DATA
  local LOGIN_COMMITS
  local NUMBERED_CONTRIBUTORS
  local NEW_SECTION
  local TEMP_FILE

  make_temp_file ALLCOMMITS
  make_temp_file CONTRIBUTORS_DATA
  make_temp_file LOGIN_COMMITS
  make_temp_file NUMBERED_CONTRIBUTORS
  make_temp_file NEW_SECTION
  make_temp_file TEMP_FILE

  # ---------------------------------------------------------------------------
  # 1. Collect full git history.
  # ---------------------------------------------------------------------------

  git log --pretty=format:'%ct|%H|%aE|%cI|%aN' \
    --reverse --all > "$ALLCOMMITS"

  # ---------------------------------------------------------------------------
  # 2. Merge commits by email, then by GitHub login.
  # ---------------------------------------------------------------------------

  collect_contributors_by_login "$ALLCOMMITS" "$CONTRIBUTORS_DATA" "$LOGIN_COMMITS"

  local CONTRIBUTOR_COUNT
  CONTRIBUTOR_COUNT=$(wc -l < "$CONTRIBUTORS_DATA" | tr -d ' ')

  echo "Found ${CONTRIBUTOR_COUNT} unique contributors"

  # ---------------------------------------------------------------------------
  # 3. Generate new Contributors section.
  # ---------------------------------------------------------------------------

  local CURRENT_DATE
  CURRENT_DATE=$(date +%Y-%m-%d)
  generate_numbered_contributors "$CONTRIBUTORS_DATA" "$NUMBERED_CONTRIBUTORS" "$CONTRIBUTOR_COUNT" "full"
  run_link_check "$NUMBERED_CONTRIBUTORS" "$LOGIN_COMMITS"

  {
    echo "<!-- last_commit: ${CURRENT_HEAD} -->"
    echo ""
    echo "Every release of vLLM Ascend would not have been possible without the following contributors:"
    echo ""
    echo "Updated on ${CURRENT_DATE}:"
    echo ""
    echo "| Number | Contributor | Date | Commit ID |"
    echo "|:------:|:-----------:|:-----:|:---------:|"
    write_contributor_rows "$NUMBERED_CONTRIBUTORS" -
  } > "$NEW_SECTION"

  # ---------------------------------------------------------------------------
  # 4. Replace or append the Contributors section in contributors.md.
  # ---------------------------------------------------------------------------

  local FOUND_CONTRIBUTORS=false

  while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" == "## Contributors" ]]; then
      FOUND_CONTRIBUTORS=true
      echo "$line" >> "$TEMP_FILE"
      cat "$NEW_SECTION" >> "$TEMP_FILE"
      break
    else
      echo "$line" >> "$TEMP_FILE"
    fi
  done < "$CONTRIBUTORS_FILE"

  if ! $FOUND_CONTRIBUTORS; then
    {
      echo ""
      echo "## Contributors"
      cat "$NEW_SECTION"
    } >> "$TEMP_FILE"
    echo ""
    echo "Warning: '## Contributors' section not found, appended at the end."
  fi

  mv "$TEMP_FILE" "$CONTRIBUTORS_FILE"

  echo "Done! Contributors list has been updated in: ${CONTRIBUTORS_FILE}"
}

# ==============================================================================
# Step 4C: Sort only
# ==============================================================================

run_sort_only() {
  echo "Refreshing contributor table numbers only..."
  echo ""

  local TEMP_FILE
  make_temp_file TEMP_FILE

  # ---------------------------------------------------------------------------
  # 1. Count existing rows in the Contributors table.
  # ---------------------------------------------------------------------------

  local ROW_COUNT=0
  local IN_TABLE=false
  local IN_ROWS=false

  while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" == "| Number | Contributor | Date | Commit ID |" ]]; then
      IN_TABLE=true
      IN_ROWS=false
      continue
    fi

    if [ "$IN_TABLE" = true ] && [[ "$line" == "|:"* ]]; then
      IN_ROWS=true
      continue
    fi

    if [ "$IN_TABLE" = true ] && [ "$IN_ROWS" = true ]; then
      if [[ "$line" =~ ^\|[[:space:]]*[0-9]+[[:space:]]*\| ]]; then
        ROW_COUNT=$((ROW_COUNT + 1))
      else
        break
      fi
    fi
  done < "$CONTRIBUTORS_FILE"

  if [ "$ROW_COUNT" -eq 0 ]; then
    die "No contributor rows found in: ${CONTRIBUTORS_FILE}"
  fi

  # ---------------------------------------------------------------------------
  # 2. Rewrite only the Number column, preserving row order and other content.
  # ---------------------------------------------------------------------------

  local NEXT_NUMBER="$ROW_COUNT"
  IN_TABLE=false
  IN_ROWS=false

  while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" == "| Number | Contributor | Date | Commit ID |" ]]; then
      IN_TABLE=true
      IN_ROWS=false
      echo "$line" >> "$TEMP_FILE"
      continue
    fi

    if [ "$IN_TABLE" = true ] && [[ "$line" == "|:"* ]]; then
      IN_ROWS=true
      echo "$line" >> "$TEMP_FILE"
      continue
    fi

    if [ "$IN_TABLE" = true ] && [ "$IN_ROWS" = true ]; then
      if [[ "$line" =~ ^\|[[:space:]]*[0-9]+[[:space:]]*\| ]]; then
        echo "$line" | sed -E "s/^\|[[:space:]]*[0-9]+[[:space:]]*\|/| ${NEXT_NUMBER} |/" >> "$TEMP_FILE"
        NEXT_NUMBER=$((NEXT_NUMBER - 1))
        continue
      fi

      IN_TABLE=false
      IN_ROWS=false
    fi

    echo "$line" >> "$TEMP_FILE"
  done < "$CONTRIBUTORS_FILE"

  mv "$TEMP_FILE" "$CONTRIBUTORS_FILE"

  echo "Done! Refreshed ${ROW_COUNT} contributor row numbers in: ${CONTRIBUTORS_FILE}"
}

# ==============================================================================
# Main
# ==============================================================================

main() {
  trap cleanup_temp_files EXIT

  parse_args "$@"
  init_context

  if [ "$SORT_ONLY" = true ]; then
    run_sort_only
    return
  fi

  determine_update_mode

  if [ "$INCREMENTAL" = true ]; then
    run_incremental_update
  else
    run_full_refresh
  fi
}

main "$@"
