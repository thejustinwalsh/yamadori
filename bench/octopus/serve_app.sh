#!/bin/bash
# The grader's app container (bench/octopus/grade.py): build a COPY of the
# model's project, then serve it on :3001 for browser_check.py.
#
#   docker run ... -v <project>:/src:ro -v <out>:/out -v <bench/octopus>:/grader:ro \
#       <node image> bash /grader/serve_app.sh ts|npm|js [build]
#
# ts   npm install, tsc --noEmit, npm run build (V1-V3: a TypeScript project)
# npm  the same, but tsc only when the project has a tsconfig.json (V4 and the
#      pagoda task: the model chose the language; grade.build_mode); without
#      one tsc.log says "skipped" and tsc.rc is "skip"
# js   served as is (V0; a V4 project with no package.json)
#
# /out/stage says where it is: building -> "serving static" | "serving dev" |
# failed. Every step's log, exit code and seconds land in /out.
#
# BUILD-ONLY (second argument `build`; bench/voxel/check.py): ts / npm run the
# same install / tsc / build steps, then dist/ is copied to /out/dist and the
# container exits: /out/stage is "built" (dist/index.html exists) or failed.
# `js build` copies the project itself (index.html at its root) to /out/dist.
# Without the argument nothing changes.
set -u
OUT=/out
MODE=${2:-serve}
mkdir -p "$OUT"
echo building > "$OUT/stage"
if [ "$1" = ts ] || [ "$1" = npm ]; then
  mkdir -p /app
  (cd /src && tar --exclude=./node_modules --exclude=./dist -cf - .) | (cd /app && tar xf -)
  cd /app
  s=$(date +%s); timeout 900 npm install --no-audit --no-fund > "$OUT/npm_install.log" 2>&1
  echo $? > "$OUT/npm_install.rc"; echo $(( $(date +%s) - s )) > "$OUT/npm_install.s"
  if [ "$1" = ts ] || [ -f tsconfig.json ]; then
    timeout 600 npx --no-install tsc --noEmit -p . > "$OUT/tsc.log" 2>&1; echo $? > "$OUT/tsc.rc"
  else
    echo "skipped: no tsconfig.json" > "$OUT/tsc.log"; echo skip > "$OUT/tsc.rc"
  fi
  s=$(date +%s); timeout 900 npm run build > "$OUT/build.log" 2>&1
  echo $? > "$OUT/build.rc"; echo $(( $(date +%s) - s )) > "$OUT/build.s"
  npm ls --depth=0 --json > "$OUT/npm_ls.json" 2> /dev/null
  if [ "$MODE" = build ]; then
    if [ -f dist/index.html ]; then
      cp -r dist "$OUT/dist" && echo built > "$OUT/stage" || echo failed > "$OUT/stage"
    else
      echo failed > "$OUT/stage"
    fi
    exit 0
  fi
  if [ -f dist/index.html ]; then
    echo "serving static" > "$OUT/stage"
    cd dist && exec python3 -m http.server 3001 --bind 0.0.0.0 > "$OUT/server.log" 2>&1
  elif [ "$(cat "$OUT/npm_install.rc")" = 0 ]; then
    echo "serving dev" > "$OUT/stage"
    exec npx --no-install vite --host 0.0.0.0 --port 3001 --strictPort > "$OUT/server.log" 2>&1
  else
    echo failed > "$OUT/stage"
    exit 0
  fi
else
  cd /src
  if [ "$MODE" = build ]; then
    if [ -f index.html ]; then
      mkdir -p "$OUT/dist" && cp -r . "$OUT/dist" && echo built > "$OUT/stage" \
        || echo failed > "$OUT/stage"
    else
      echo failed > "$OUT/stage"
    fi
    exit 0
  fi
  echo "serving static" > "$OUT/stage"
  exec python3 -m http.server 3001 --bind 0.0.0.0 > "$OUT/server.log" 2>&1
fi
