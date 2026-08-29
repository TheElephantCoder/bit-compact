use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::storage::CompactReader;

fn url_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '%' {
            let hi = chars.next().unwrap_or('0');
            let lo = chars.next().unwrap_or('0');
            let hex = format!("{hi}{lo}");
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                out.push(byte as char);
            }
        } else if c == '+' {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

fn parse_query(query: Option<&str>) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if let Some(q) = query {
        for pair in q.split('&') {
            if pair.is_empty() { continue; }
            let mut split = pair.splitn(2, '=');
            let k = split.next().unwrap_or("");
            let v = split.next().unwrap_or("");
            map.insert(url_decode(k), url_decode(v));
        }
    }
    map
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            _ => out.push(c),
        }
    }
    out
}

fn json_error(msg: &str) -> String {
    format!("{{\"error\":\"{}\"}}", json_escape(msg))
}

fn hex_short(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(16);
    for b in &bytes[..8] {
        s.push_str(&format!("{b:02x}"));
    }
    s.push_str("…");
    s
}

fn write_response(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
}

fn write_json(stream: &mut TcpStream, status: &str, body: &str) {
    write_response(stream, status, "application/json; charset=utf-8", body.as_bytes());
}

fn open_reader(path: &Path) -> Result<CompactReader, String> {
    CompactReader::open(path).map_err(|e| e.to_string())
}

fn handle_api_info(params: &HashMap<String, String>, default_file: Option<&Path>) -> (String, String) {
    let file = params.get("file").map(|s| PathBuf::from(s)).or_else(|| default_file.map(|p| p.to_path_buf()));
    let p = match file {
        Some(f) => f,
        None => return ("400 Bad Request".into(), json_error("no file specified; start server with `bitcompact serve <file>` or use ?file=path")),
    };
    match open_reader(&p) {
        Ok(r) => {
            // include actual file path in json
            let h = r.header();
            let body = format!(
                "{{\"path\":\"{}\",\"dims\":{},\"count\":{},\"quant\":\"{:?}\",\"distance\":\"{:?}\",\"major\":{},\"minor\":{},\"footer_offset\":{},\"checksum\":\"{}\",\"file_size\":{}}}",
                json_escape(&p.display().to_string()),
                r.dims(),
                r.len(),
                r.quant_type(),
                r.distance_metric(),
                h.major,
                h.minor,
                r.footer_offset(),
                hex_short(r.checksum()),
                std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0)
            );
            ("200 OK".into(), body)
        }
        Err(e) => ("500 Internal Server Error".into(), json_error(&e)),
    }
}

fn handle_api_get(params: &HashMap<String, String>, default_file: Option<&Path>) -> (String, String) {
    let file = params.get("file").map(|s| PathBuf::from(s)).or_else(|| default_file.map(|p| p.to_path_buf()));
    let p = match file {
        Some(f) => f,
        None => return ("400 Bad Request".into(), json_error("no file")),
    };
    let id_str = match params.get("id") {
        Some(v) => v,
        None => return ("400 Bad Request".into(), json_error("missing id")),
    };
    let id: u64 = match id_str.parse() {
        Ok(v) => v,
        Err(_) => return ("400 Bad Request".into(), json_error("invalid id")),
    };
    match open_reader(&p) {
        Ok(r) => {
            if id >= r.len() {
                return ("404 Not Found".into(), json_error(&format!("id {id} out of range {}", r.len())));
            }
            match (r.get_quantized(id), r.get_vector(id)) {
                (Ok(q), Ok(v)) => {
                    let q_str = q.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(",");
                    let v_str = v.iter().map(|f| format!("{:.6}", f)).collect::<Vec<_>>().join(",");
                    let body = format!("{{\"id\":{id},\"quantized\":[{q_str}],\"vector\":[{v_str}]}}");
                    ("200 OK".into(), body)
                }
                (Err(e), _) | (_, Err(e)) => ("500 Internal Server Error".into(), json_error(&e.to_string())),
            }
        }
        Err(e) => ("500 Internal Server Error".into(), json_error(&e)),
    }
}

fn handle_api_search(params: &HashMap<String, String>, default_file: Option<&Path>) -> (String, String) {
    let file = params.get("file").map(|s| PathBuf::from(s)).or_else(|| default_file.map(|p| p.to_path_buf()));
    let p = match file {
        Some(f) => f,
        None => return ("400 Bad Request".into(), json_error("no file")),
    };
    let query_str = match params.get("query") {
        Some(v) => v,
        None => return ("400 Bad Request".into(), json_error("missing query (comma-separated floats)")),
    };
    let k: usize = params.get("k").and_then(|s| s.parse().ok()).unwrap_or(5);
    let query: Vec<f32> = query_str.split(',').filter_map(|s| s.trim().parse::<f32>().ok()).collect();
    if query.is_empty() {
        return ("400 Bad Request".into(), json_error("query empty or not floats"));
    }
    match open_reader(&p) {
        Ok(r) => {
            if query.len() != r.dims() {
                return ("400 Bad Request".into(), json_error(&format!("query dims {} != file dims {}", query.len(), r.dims())));
            }
            match r.search(&query, k) {
                Ok(hits) => {
                    let hits_json = hits.iter().map(|h| format!("{{\"id\":{},\"distance\":{:.6}}}", h.id, h.distance)).collect::<Vec<_>>().join(",");
                    let body = format!("{{\"query\":[{}],\"k\":{k},\"hits\":[{hits_json}]}}", query.iter().map(|f| format!("{:.6}", f)).collect::<Vec<_>>().join(","));
                    ("200 OK".into(), body)
                }
                Err(e) => ("500 Internal Server Error".into(), json_error(&e.to_string())),
            }
        }
        Err(e) => ("500 Internal Server Error".into(), json_error(&e)),
    }
}

fn handle_api_validate(params: &HashMap<String, String>, default_file: Option<&Path>) -> (String, String) {
    let file = params.get("file").map(|s| PathBuf::from(s)).or_else(|| default_file.map(|p| p.to_path_buf()));
    let p = match file {
        Some(f) => f,
        None => return ("400 Bad Request".into(), json_error("no file")),
    };
    match crate::validate::validate(&p) {
        Ok(rep) => {
            let warnings = rep.warnings.iter().map(|w| format!("\"{}\"", json_escape(w))).collect::<Vec<_>>().join(",");
            let body = format!(
                "{{\"path\":\"{}\",\"dims\":{},\"count\":{},\"footer_offset\":{},\"file_size\":{},\"checksum_valid\":{},\"row_ids_monotonic\":{},\"metadata_finite\":{},\"is_valid\":{},\"warnings\":[{}]}}",
                json_escape(&rep.path),
                rep.dims,
                rep.count,
                rep.footer_offset,
                rep.file_size,
                rep.checksum_valid,
                rep.row_ids_monotonic,
                rep.metadata_finite,
                rep.is_valid(),
                warnings
            );
            ("200 OK".into(), body)
        }
        Err(e) => ("500 Internal Server Error".into(), json_error(&e.to_string())),
    }
}

fn handle_api_stats(params: &HashMap<String, String>, default_file: Option<&Path>) -> (String, String) {
    let file = params.get("file").map(|s| PathBuf::from(s)).or_else(|| default_file.map(|p| p.to_path_buf()));
    let p = match file {
        Some(f) => f,
        None => return ("400 Bad Request".into(), json_error("no file")),
    };
    match open_reader(&p) {
        Ok(r) => {
            let file_size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            let data_bytes = r.len() * r.dims() as u64;
            let original = data_bytes * 4;
            let ratio = if data_bytes == 0 { 0.0 } else { original as f64 / data_bytes as f64 };
            let body = format!(
                "{{\"path\":\"{}\",\"dims\":{},\"count\":{},\"file_size\":{},\"data_bytes\":{},\"original_bytes\":{},\"ratio\":{:.2}}}",
                json_escape(&p.display().to_string()),
                r.dims(),
                r.len(),
                file_size,
                data_bytes,
                original,
                ratio
            );
            ("200 OK".into(), body)
        }
        Err(e) => ("500 Internal Server Error".into(), json_error(&e)),
    }
}

fn handle_client(mut stream: TcpStream, default_file: Option<PathBuf>) {
    let mut buf = [0u8; 8192];
    let n = match stream.read(&mut buf) {
        Ok(0) | Err(_) => return,
        Ok(v) => v,
    };
    let req = String::from_utf8_lossy(&buf[..n]);
    let mut lines = req.lines();
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path_full = parts.next().unwrap_or("/");
    if method != "GET" {
        write_json(&mut stream, "405 Method Not Allowed", &json_error("only GET"));
        return;
    }
    let (path, query) = match path_full.find('?') {
        Some(idx) => (&path_full[..idx], Some(&path_full[idx + 1..])),
        None => (path_full, None),
    };
    let params = parse_query(query);
    let default_ref = default_file.as_deref();

    match path {
        "/" | "/index.html" => {
            write_response(&mut stream, "200 OK", "text/html; charset=utf-8", GUI_HTML.as_bytes());
        }
        "/style.css" => {
            write_response(&mut stream, "200 OK", "text/css; charset=utf-8", GUI_CSS.as_bytes());
        }
        "/api/info" => {
            let (status, body) = handle_api_info(&params, default_ref);
            write_json(&mut stream, &status, &body);
        }
        "/api/get" => {
            let (status, body) = handle_api_get(&params, default_ref);
            write_json(&mut stream, &status, &body);
        }
        "/api/search" => {
            let (status, body) = handle_api_search(&params, default_ref);
            write_json(&mut stream, &status, &body);
        }
        "/api/validate" => {
            let (status, body) = handle_api_validate(&params, default_ref);
            write_json(&mut stream, &status, &body);
        }
        "/api/stats" => {
            let (status, body) = handle_api_stats(&params, default_ref);
            write_json(&mut stream, &status, &body);
        }
        "/api/health" => {
            write_json(&mut stream, "200 OK", "{\"status\":\"ok\"}");
        }
        _ => {
            write_json(&mut stream, "404 Not Found", &json_error("not found"));
        }
    }
}

pub fn serve(file: Option<PathBuf>, host: &str, port: u16) -> std::io::Result<()> {
    let addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&addr)?;
    let default_file = file.map(|p| Arc::new(p));
    println!("bitcompact serve — GUI at http://{addr}/");
    if let Some(f) = &default_file {
        println!("  file: {}", f.display());
    } else {
        println!("  no file preloaded — use ?file=path in API or open via GUI");
    }
    println!("  press Ctrl+C to stop");

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let df = default_file.clone().map(|a| (*a).clone());
                std::thread::spawn(move || handle_client(s, df));
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}

const GUI_CSS: &str = r#"
:root{--bg:#0a0a0b;--fg:#e8e8e8;--muted:#9aa0a6;--accent:#6ee7b7;--accent2:#38bdf8;--card:#141416;--border:#222227;--code:#1a1a1e}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
a{color:var(--accent2)}header{padding:20px 24px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
header h1{margin:0;font-size:20px;letter-spacing:-0.02em}header h1 span{color:var(--accent)}
.container{max-width:1200px;margin:0 auto;padding:20px 24px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
.card h3{margin:0 0 8px;font-size:14px;letter-spacing:0.02em;text-transform:uppercase;color:var(--muted)}
input,button{font:14px ui-sans-serif;padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:var(--code);color:var(--fg)}
button{background:var(--fg);color:var(--bg);font-weight:600;cursor:pointer}
button.ghost{background:transparent;color:var(--fg)}
input{width:100%}pre{background:var(--code);border:1px solid var(--border);border-radius:8px;padding:12px;overflow:auto;font:12px ui-monospace,monospace;max-height:300px}
.badge{font:11px ui-monospace,monospace;padding:4px 8px;border:1px solid var(--border);border-radius:999px;background:var(--card);color:var(--muted)}
canvas{width:100%;height:140px;background:var(--code);border:1px solid var(--border);border-radius:8px}
.kv{display:grid;grid-template-columns:120px 1fr;gap:6px;font-size:13px}.kv div:nth-child(odd){color:var(--muted)}
"#;

const GUI_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>bitcompact — serve GUI</title>
<link rel="stylesheet" href="/style.css">
</head>
<body>
<header>
  <h1>bit<span>-compact</span> <small style="font-weight:400;color:var(--muted);margin-left:8px;">serve</small></h1>
  <div><span class="badge">1 seek · 0 alloc</span> <span class="badge">BTCP 32B</span> <span class="badge">SHA-256</span></div>
</header>
<div class="container">
  <div class="grid">
    <div class="card">
      <h3>File</h3>
      <div class="kv" id="info">loading…</div>
      <div style="display:flex;gap:8px;margin-top:12px">
        <input id="fileInput" placeholder="/tmp/vectors.btcp (or leave empty for preloaded)">
        <button onclick="loadInfo()">Load</button>
        <button class="ghost" onclick="doValidate()">Validate</button>
      </div>
      <pre id="validateOut" style="display:none"></pre>
    </div>
    <div class="card">
      <h3>Stats</h3>
      <div class="kv" id="stats">—</div>
      <button class="ghost" onclick="loadStats()" style="margin-top:12px">Refresh stats</button>
    </div>
  </div>

  <div class="grid" style="margin-top:16px">
    <div class="card">
      <h3>Browse vector</h3>
      <div style="display:flex;gap:8px">
        <input id="idInput" type="number" min="0" value="0" placeholder="id">
        <button onclick="loadVector()">Get</button>
        <button class="ghost" onclick="prevVec()">◀</button>
        <button class="ghost" onclick="nextVec()">▶</button>
      </div>
      <canvas id="chart" width="600" height="140"></canvas>
      <pre id="vecOut">—</pre>
    </div>
    <div class="card">
      <h3>Search (top-k)</h3>
      <input id="queryInput" placeholder="query as comma-separated, e.g. 0.1,0.2,0.3">
      <div style="display:flex;gap:8px;margin-top:8px">
        <input id="kInput" type="number" min="1" max="100" value="5" style="width:80px">
        <button onclick="doSearch()">Search</button>
      </div>
      <pre id="searchOut">—</pre>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h3>Log</h3>
    <pre id="log">GUI ready. API: /api/info, /api/get?id=, /api/search?query=&k=, /api/validate, /api/stats, /api/health</pre>
  </div>
</div>
<script>
let dims = 0, count = 0;
let currentId = 0;
function fileParam(){ const v=document.getElementById('fileInput').value.trim(); return v? `&file=${encodeURIComponent(v)}` : ''; }
function fileQuery(){ const v=document.getElementById('fileInput').value.trim(); return v? `?file=${encodeURIComponent(v)}` : ''; }
async function fetchJSON(url){
  const r=await fetch(url);
  const t=await r.text();
  try { return JSON.parse(t); } catch { return {error:t, status:r.status}; }
}
async function loadInfo(){
  const q=fileQuery();
  const j=await fetchJSON('/api/info'+q);
  const el=document.getElementById('info');
  if(j.error){ el.innerHTML=`<div style="color:#f87171">${j.error}</div>`; log(j.error); return; }
  dims=j.dims; count=j.count;
  el.innerHTML=`<div>path</div><div>${j.path}</div><div>dims</div><div>${j.dims}</div><div>count</div><div>${j.count}</div><div>quant</div><div>${j.quant}</div><div>distance</div><div>${j.distance}</div><div>footer</div><div>${j.footer_offset}</div><div>checksum</div><div>${j.checksum}</div><div>size</div><div>${j.file_size} B</div>`;
  log(`info dims=${j.dims} count=${j.count}`);
  document.getElementById('queryInput').placeholder = Array(dims).fill('0.0').join(',');
  loadStats();
}
async function loadStats(){
  const q=fileQuery();
  const j=await fetchJSON('/api/stats'+q);
  const el=document.getElementById('stats');
  if(j.error){ el.textContent=j.error; return; }
  el.innerHTML=`<div>file</div><div>${j.path}</div><div>dims × count</div><div>${j.dims} × ${j.count}</div><div>data</div><div>${j.data_bytes} B</div><div>original</div><div>${j.original_bytes} B</div><div>ratio</div><div>${j.ratio}×</div><div>file size</div><div>${j.file_size} B</div>`;
}
async function loadVector(){
  const id=parseInt(document.getElementById('idInput').value||'0',10);
  currentId=isNaN(id)?0:id;
  const q=fileParam();
  const j=await fetchJSON(`/api/get?id=${currentId}${q}`);
  const el=document.getElementById('vecOut');
  if(j.error){ el.textContent=j.error; return; }
  el.textContent=`id ${j.id}\nquantized [${j.quantized.slice(0,16).join(',')}${j.quantized.length>16?' …':''}]\nvector [${j.vector.slice(0,8).map(v=>Number(v).toFixed(4)).join(', ')}${j.vector.length>8?' …':''}]`;
  drawChart(j.vector);
  log(`get id=${j.id} ok`);
}
function prevVec(){ let v=parseInt(document.getElementById('idInput').value||'0',10)-1; if(v<0) v=0; document.getElementById('idInput').value=v; loadVector(); }
function nextVec(){ let v=parseInt(document.getElementById('idInput').value||'0',10)+1; if(v>=count) v=count-1; document.getElementById('idInput').value=v; loadVector(); }
async function doSearch(){
  const q=document.getElementById('queryInput').value.trim();
  const k=document.getElementById('kInput').value||'5';
  if(!q){ alert('enter query as comma-separated floats'); return; }
  const file=fileParam();
  const j=await fetchJSON(`/api/search?query=${encodeURIComponent(q)}&k=${k}${file}`);
  const el=document.getElementById('searchOut');
  if(j.error){ el.textContent=j.error; return; }
  el.textContent=j.hits.map(h=>`id=${h.id} dist=${Number(h.distance).toFixed(6)}`).join('\n') || 'no hits';
  log(`search k=${j.k} hits=${j.hits.length}`);
}
async function doValidate(){
  const q=fileQuery();
  const j=await fetchJSON('/api/validate'+q);
  const el=document.getElementById('validateOut');
  el.style.display='block';
  el.textContent=JSON.stringify(j,null,2);
}
function drawChart(vec){
  const c=document.getElementById('chart');
  const ctx=c.getContext('2d');
  const w=c.width, h=c.height;
  ctx.clearRect(0,0,w,h);
  if(!vec || !vec.length) return;
  const min=Math.min(...vec), max=Math.max(...vec);
  const range=(max-min)||1;
  ctx.strokeStyle='#6ee7b7'; ctx.lineWidth=1.5; ctx.beginPath();
  vec.forEach((v,i)=>{
    const x= (i/(vec.length-1))*w;
    const y= h - ((v-min)/range)*h*0.9 - h*0.05;
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();
  ctx.fillStyle='#38bdf8';
  vec.forEach((v,i)=>{
    const x= (i/(vec.length-1))*w;
    const y= h - ((v-min)/range)*h*0.9 - h*0.05;
    ctx.beginPath(); ctx.arc(x,y,2,0,Math.PI*2); ctx.fill();
  });
}
function log(m){ const el=document.getElementById('log'); el.textContent=`[${new Date().toLocaleTimeString()}] ${m}\n`+el.textContent; }
loadInfo();
</script>
</body>
</html>
"#;
