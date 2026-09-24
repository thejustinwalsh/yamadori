#!/bin/bash
# The grader's app container (bench/octopus/grade.py): build a COPY of the
# model's project, then serve it on :3001 for browser_check.py.
#
#   docker run ... -v <project>:/src:ro -v <out>:/out -v <bench/octopus>:/grader:ro \
#       <node image> bash /grader/serve_app.sh ts|js
#
# /out/stage says where it is: building -> "serving static" | "serving dev" |
# failed. Every step's log, exit code and seconds land in /out.
set -u
OUT=/out
mkdir -p "$OUT"
echo building > "$OUT/stage"
if [ "$1" = ts ]; then
  mkdir -p /app
  (cd /src && tar --exclude=./node_modules --exclude=./dist -cf - .) | (cd /app && tar xf -)
  cd /app
  s=$(date +%s); timeout 900 npm install --no-audit --no-fund > "$OUT/npm_install.log" 2>&1
  echo $? > "$OUT/npm_install.rc"; echo $(( $(date +%s) - s )) > "$OUT/npm_install.s"
  timeout 600 npx --no-install tsc --noEmit -p . > "$OUT/tsc.log" 2>&1; echo $? > "$OUT/tsc.rc"
  s=$(date +%s); timeout 900 npm run build > "$OUT/build.log" 2>&1
  echo $? > "$OUT/build.rc"; echo $(( $(date +%s) - s )) > "$OUT/build.s"
  npm ls --depth=0 --json > "$OUT/npm_ls.json" 2> /dev/null
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
  echo "serving static" > "$OUT/stage"
  exec python3 -m http.server 3001 --bind 0.0.0.0 > "$OUT/server.log" 2>&1
fi
