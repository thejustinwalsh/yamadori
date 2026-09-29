#!/usr/bin/env node
// Print an installed package's API from its own type declarations, the way an
// editor's hover shows it: the exported names with their signatures, or one
// name in full (every overload, the members of a type, the doc comment, the
// file and line it is declared at). Read-only; no network.
//
//   node api.cjs <module>                 every export, one line each
//   node api.cjs <module> <name> [...]    those names in full
//   node api.cjs <module> 'Instanced*'    the exports whose names match
//
// <module> is what the code imports: koota, koota/react, @react-three/fiber,
// three. Run it from the project root: modules resolve the way the project's
// tsconfig.json resolves them (moduleResolution bundler when it has none).
// TypeScript comes from the project, else the one installed with the harness.
"use strict";
const path = require("path");

function loadTs(cwd) {
  // the project's own, the harness box's, the Octopus terminal's tools volume
  for (const base of [cwd, "/opt/harness/node_modules"]) {
    try { return require(require.resolve("typescript", { paths: [base] })); } catch (_) { /* next */ }
  }
  try { return require("/opt/yamadori-tools/typescript"); } catch (_) { /* none */ }
  console.error("typescript not found (the project has none and none is installed with the harness)");
  process.exit(2);
}

const argv = process.argv.slice(2);
if (!argv.length || argv[0] === "-h" || argv[0] === "--help") {
  console.log("usage: node api.cjs <module> [name|glob ...]   (run from the project root)");
  process.exit(argv.length ? 0 : 2);
}
const [mod, ...names] = argv;
const cwd = process.cwd();
const ts = loadTs(cwd);

let options = {
  target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext,
  moduleResolution: ts.ModuleResolutionKind.Bundler, jsx: ts.JsxEmit.ReactJSX,
  esModuleInterop: true, allowSyntheticDefaultImports: true, skipLibCheck: true, strict: true,
};
const cfgPath = ts.findConfigFile(cwd, ts.sys.fileExists, "tsconfig.json");
if (cfgPath) {
  const cfg = ts.readConfigFile(cfgPath, ts.sys.readFile);
  if (!cfg.error) {
    const parsed = ts.parseJsonConfigFileContent(cfg.config, ts.sys, path.dirname(cfgPath));
    options = { ...options, ...parsed.options };
  }
}
options = { ...options, noEmit: true, skipLibCheck: true, allowJs: true, checkJs: false };

const probe = path.join(cwd, "__package_api_probe__.ts");
// The package's other entry points (package.json "exports"), when <module> is
// the package itself: listed, and searched for a name the root does not export.
function subpaths(spec) {
  const parts = spec.split("/"), pkg = parts[0].startsWith("@") ? parts.slice(0, 2).join("/") : parts[0];
  if (spec !== pkg) return [];
  for (let d = cwd; ; d = path.dirname(d)) {
    const pj = path.join(d, "node_modules", pkg, "package.json");
    if (ts.sys.fileExists(pj)) {
      let ex;
      try { ex = JSON.parse(ts.sys.readFile(pj)).exports; } catch (_) { return []; }
      if (!ex || typeof ex !== "object" || Array.isArray(ex)) return [];
      return Object.keys(ex).filter((k) => k.startsWith("./") && k !== "./package.json" && !k.includes("*")
        && !/\.(css|json)$/.test(k)).map((k) => pkg + k.slice(1));
    }
    if (path.dirname(d) === d) return [];
  }
}
const subs = subpaths(mod);
const probeText = [mod, ...subs].map((m, i) => `import * as __m${i} from ${JSON.stringify(m)};`).join("\n")
  + "\nexport {};\n";
const host = ts.createCompilerHost(options);
const readFile = host.readFile, fileExists = host.fileExists, getSourceFile = host.getSourceFile;
host.fileExists = (f) => path.resolve(f) === probe || fileExists.call(host, f);
host.readFile = (f) => (path.resolve(f) === probe ? probeText : readFile.call(host, f));
host.getSourceFile = (f, lang, ...rest) => (path.resolve(f) === probe
  ? ts.createSourceFile(f, probeText, lang, true) : getSourceFile.call(host, f, lang, ...rest));
const program = ts.createProgram([probe], options, host);
const checker = program.getTypeChecker();
const sf = program.getSourceFile(probe);
const modSym = checker.getSymbolAtLocation(sf.statements[0].moduleSpecifier);
const subSyms = subs.map((m, i) => [m, checker.getSymbolAtLocation(sf.statements[i + 1].moduleSpecifier)])
  .filter(([, x]) => x);
if (!modSym) {
  const r = ts.resolveModuleName(mod, probe, options, ts.sys);
  const found = r.resolvedModule ? r.resolvedModule.resolvedFileName : null;
  console.error(found
    ? `${mod}: resolved to ${rel(found)}, which declares no types`
    : `${mod}: not found from ${cwd} (is it installed? run this from the project root)`);
  process.exit(1);
}

const F = ts.TypeFormatFlags.NoTruncation;
function rel(f) { const r = path.relative(cwd, f); return r.startsWith("..") ? f : r; }
function where(sym) {
  const d = (sym.declarations || [])[0];
  if (!d) return "";
  const s = d.getSourceFile();
  const { line } = s.getLineAndCharacterOfPosition(d.getStart());
  return `${rel(s.fileName)}:${line + 1}`;
}
function doc(sym) {
  const t = ts.displayPartsToString(sym.getDocumentationComment(checker)).trim();
  return t;
}
function resolve(sym) {
  return sym.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(sym) : sym;
}
function kindOf(s) {
  const f = s.flags, S = ts.SymbolFlags;
  if (f & S.Class) return "class";
  if (f & S.Function) return "function";
  if (f & S.Interface) return "interface";
  if (f & S.TypeAlias) return "type";
  if (f & S.Enum) return "enum";
  if (f & S.Variable) return "const";
  if (f & S.Module) return "namespace";
  if (f & (S.Property | S.Method)) return "member";
  return "export";
}
function tparams(sym) {
  const d = (sym.declarations || []).find((x) => x.typeParameters && x.typeParameters.length);
  return d ? "<" + d.typeParameters.map((t) => t.getText()).join(", ") + ">" : "";
}
function sigs(type) { return type.getCallSignatures(); }
function sigText(sig) { return checker.signatureToString(sig, undefined, F); }
function short(s, n) { s = s.replace(/\s+/g, " "); return s.length > n ? s.slice(0, n - 3) + "..." : s; }

function oneLine(name, sym) {
  const s = resolve(sym), k = kindOf(s);
  let text = "";
  if (k === "function") {
    const ss = sigs(checker.getTypeOfSymbol(s));
    text = sigText(ss[0] || {}) + (ss.length > 1 ? `  (+${ss.length - 1} overload${ss.length > 2 ? "s" : ""})` : "");
  } else if (k === "const") {
    text = ": " + checker.typeToString(checker.getTypeOfSymbol(s), undefined, F);
  } else if (k === "type") {
    text = " = " + checker.typeToString(checker.getDeclaredTypeOfSymbol(s), undefined, F | ts.TypeFormatFlags.InTypeAlias);
  } else if (k === "namespace") {
    text = ` (${checker.getExportsOfModule(s).length} exports)`;
  } else if (k === "class") {
    const c = checker.getTypeOfSymbol(s).getConstructSignatures()[0];
    text = c ? " constructor" + sigText(c).replace(/:\s*[^:]*$/, "") : "";
  }
  return short(`${k} ${name}${k === "type" || k === "interface" ? tparams(s) : ""}${text}`, 200);
}

function pkgOf(file) {
  if (program.isSourceFileDefaultLibrary(file)) return "the TypeScript lib";
  const m = file.fileName.split(path.sep).join("/").match(/.*node_modules\/((?:@[^/]+\/)?[^/]+)/);
  return m ? m[1] : "the project";
}
// Members declared by the package that declares the type are listed; the ones
// it inherits from another package (React's DOM props, Array's methods) are
// counted, by package.
function members(type, indent, owner) {
  const out = [], other = new Map();
  const home = owner && owner.declarations && owner.declarations[0]
    ? pkgOf(owner.declarations[0].getSourceFile()) : null;
  for (const p of checker.getPropertiesOfType(type)) {
    if (p.name.startsWith("__")) continue;
    const pd = (p.declarations || [])[0];
    const from = pd ? pkgOf(pd.getSourceFile()) : home;
    if (home && from !== home) { other.set(from, (other.get(from) || 0) + 1); continue; }
    const decl = (p.declarations || [])[0];
    const pt = decl ? checker.getTypeOfSymbolAtLocation(p, decl) : checker.getTypeOfSymbol(p);
    const opt = p.flags & ts.SymbolFlags.Optional ? "?" : "";
    const cs = sigs(pt);
    const d = doc(p).split("\n")[0];
    if ((p.flags & ts.SymbolFlags.Method) && cs.length) {
      for (const c of cs) out.push(`${indent}${p.name}${opt}${sigText(c)}`);
    } else {
      out.push(`${indent}${p.name}${opt}: ${checker.typeToString(pt, undefined, F)}`);
    }
    if (d) out[out.length - 1] += `    // ${short(d, 120)}`;
  }
  for (const [from, n] of other) out.push(`${indent}(+${n} member${n > 1 ? "s" : ""} from ${from})`);
  return out;
}

function full(name, sym) {
  const s = resolve(sym), k = kindOf(s);
  const lines = [`${k} ${name}${k === "type" || k === "interface" || k === "class" ? tparams(s) : ""}    (${where(s)})`];
  const d = doc(s);
  if (d) lines.push(...d.split("\n").map((l) => "  // " + l));
  if (k === "function" || k === "const" || k === "member") {
    const t = checker.getTypeOfSymbol(s);
    const cs = sigs(t);
    for (const c of cs) {
      lines.push(`  ${name}${sigText(c)}`);
      const cd = ts.displayPartsToString(c.getDocumentationComment(checker)).trim();
      if (cd && cd !== d) lines.push(...cd.split("\n").map((l) => "    // " + l));
    }
    if (!cs.length) lines.push(`  ${name}: ${checker.typeToString(t, undefined, F)}`);
    if (!cs.length && t.getProperties().length) lines.push(...members(t, "    ", s));
  }
  if (k === "class") {
    const st = checker.getTypeOfSymbol(s);
    for (const c of st.getConstructSignatures()) lines.push(`  new ${name}${sigText(c).replace(/:\s*[^:]*$/, "")}`);
    lines.push(...members(checker.getDeclaredTypeOfSymbol(s), "  ", s));
  }
  if (k === "interface" || k === "type") {
    const t = checker.getDeclaredTypeOfSymbol(s);
    const objectLike = t.getProperties().length && (k === "interface" || t.isIntersection() || t.flags & ts.TypeFlags.Object);
    const text = checker.typeToString(t, undefined, F | ts.TypeFormatFlags.InTypeAlias);
    if (!objectLike || t.isIntersection()) lines.push(`  = ${objectLike ? short(text, 300) : text}`);
    if (objectLike) lines.push(...members(t, "  ", s));
  }
  if (k === "namespace") {
    for (const e of checker.getExportsOfModule(s).slice().sort((a, b) => a.name.localeCompare(b.name))) {
      lines.push("  " + oneLine(e.name, e));
    }
  }
  if (k === "enum") lines.push(...(s.exports ? [...s.exports.keys()].map((m) => `  ${m}`) : []));
  return lines.join("\n");
}

const exps = checker.getExportsOfModule(modSym).slice().sort((a, b) => a.name.localeCompare(b.name));
const byName = new Map(exps.map((e) => [e.name, e]));
const glob = (g) => new RegExp("^" + g.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*").replace(/\?/g, ".") + "$");

if (!names.length) {
  console.log(`${mod}: ${exps.length} exports    (${rel(modSym.declarations[0].getSourceFile().fileName)})`);
  for (const e of exps) console.log(oneLine(e.name, e));
  if (subSyms.length) console.log(`other entry points: ${subSyms.map(([m]) => m).join(", ")}`);
  process.exit(0);
}
let missing = 0;
for (const n of names) {
  if (/[*?]/.test(n)) {
    const re = glob(n), hits = exps.filter((e) => re.test(e.name));
    if (!hits.length) { console.log(`${n}: no export of ${mod} matches`); missing++; }
    for (const e of hits) console.log(oneLine(e.name, e));
    continue;
  }
  const [head, ...rest] = n.split(".");
  let sym = byName.get(head);
  if (!sym) {
    const where = subSyms.filter(([, x]) => checker.getExportsOfModule(x).some((e) => e.name === head));
    if (where.length) {
      console.log(`${head} is exported by ${where.map(([m]) => m).join(", ")}, not by ${mod}:`);
      sym = checker.getExportsOfModule(where[0][1]).find((e) => e.name === head);
    }
  }
  if (!sym && !rest.length) {
    // one level into the namespaces (math's vec3.add, quat.fromEuler, ...)
    const found = [];
    for (const [m, ms] of [[mod, modSym], ...subSyms]) {
      for (const e of checker.getExportsOfModule(ms)) {
        const r = resolve(e);
        if (!(r.flags & (ts.SymbolFlags.ValueModule | ts.SymbolFlags.NamespaceModule))) continue;
        const hit = checker.getExportsOfModule(r).find((x) => x.name === head);
        if (hit && !found.some(([, , h]) => resolve(h) === resolve(hit))) {
          found.push([`${m} ${e.name}.${head}`, `${e.name}.${head}`, hit]);
        }
      }
    }
    if (found.length) {
      console.log(`${head} is in a namespace: ${found.map(([w]) => w).join(", ")}`);
      for (const [, label, hit] of found.slice(0, 3)) console.log(full(label, hit) + "\n");
      continue;
    }
  }
  if (!sym) {
    const near = exps.map((e) => e.name).filter((x) => x.toLowerCase().includes(head.toLowerCase().slice(0, 4))).slice(0, 8);
    console.log(`${n}: ${mod} exports no ${head}` + (near.length ? `; near: ${near.join(", ")}` : ""));
    missing++;
    continue;
  }
  let label = head;
  for (const m of rest) {
    const s = resolve(sym);
    let p;
    if (s.flags & (ts.SymbolFlags.ValueModule | ts.SymbolFlags.NamespaceModule)) {
      p = checker.getExportsOfModule(s).find((e) => e.name === m);
    } else {
      const t = s.flags & (ts.SymbolFlags.Interface | ts.SymbolFlags.TypeAlias | ts.SymbolFlags.Class)
        ? checker.getDeclaredTypeOfSymbol(s) : checker.getTypeOfSymbol(s);
      p = checker.getPropertyOfType(t, m);
    }
    if (!p) { console.log(`${label}.${m}: no such member`); sym = null; missing++; break; }
    sym = p; label += "." + m;
  }
  if (sym) console.log(full(label, sym) + "\n");
}
process.exit(missing ? 1 : 0);
