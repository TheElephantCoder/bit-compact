use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::storage::CompactReader;
use crate::transform::Transform;

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
            if pair.is_empty() {
                continue;
            }
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
    write_response(
        stream,
        status,
        "application/json; charset=utf-8",
        body.as_bytes(),
    );
}

fn open_reader(path: &Path) -> Result<CompactReader, String> {
    CompactReader::open(path).map_err(|e| e.to_string())
}

fn handle_api_info(
    params: &HashMap<String, String>,
    default_file: Option<&Path>,
) -> (String, String) {
    let file = params
        .get("file")
        .map(|s| PathBuf::from(s))
        .or_else(|| default_file.map(|p| p.to_path_buf()));
    let p = match file {
        Some(f) => f,
        None => return (
            "400 Bad Request".into(),
            json_error(
                "no file specified; start server with `bitcompact serve <file>` or use ?file=path",
            ),
        ),
    };
    match open_reader(&p) {
        Ok(r) => {
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

fn handle_api_get(
    params: &HashMap<String, String>,
    default_file: Option<&Path>,
) -> (String, String) {
    let file = params
        .get("file")
        .map(|s| PathBuf::from(s))
        .or_else(|| default_file.map(|p| p.to_path_buf()));
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
                return (
                    "404 Not Found".into(),
                    json_error(&format!("id {id} out of range {}", r.len())),
                );
            }
            match (r.get_quantized(id), r.get_vector(id)) {
                (Ok(q), Ok(v)) => {
                    let q_str = q
                        .iter()
                        .map(|b| b.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    let v_str = v
                        .iter()
                        .map(|f| format!("{:.6}", f))
                        .collect::<Vec<_>>()
                        .join(",");
                    let body =
                        format!("{{\"id\":{id},\"quantized\":[{q_str}],\"vector\":[{v_str}]}}");
                    ("200 OK".into(), body)
                }
                (Err(e), _) | (_, Err(e)) => (
                    "500 Internal Server Error".into(),
                    json_error(&e.to_string()),
                ),
            }
        }
        Err(e) => ("500 Internal Server Error".into(), json_error(&e)),
    }
}

fn handle_api_search(
    params: &HashMap<String, String>,
    default_file: Option<&Path>,
) -> (String, String) {
    let file = params
        .get("file")
        .map(|s| PathBuf::from(s))
        .or_else(|| default_file.map(|p| p.to_path_buf()));
    let p = match file {
        Some(f) => f,
        None => return ("400 Bad Request".into(), json_error("no file")),
    };
    let query_str = match params.get("query") {
        Some(v) => v,
        None => {
            return (
                "400 Bad Request".into(),
                json_error("missing query (comma-separated floats)"),
            )
        }
    };
    let k: usize = params.get("k").and_then(|s| s.parse().ok()).unwrap_or(5);
    let query: Vec<f32> = query_str
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();
    if query.is_empty() {
        return (
            "400 Bad Request".into(),
            json_error("query empty or not floats"),
        );
    }
    match open_reader(&p) {
        Ok(r) => {
            if query.len() != r.dims() {
                return (
                    "400 Bad Request".into(),
                    json_error(&format!(
                        "query dims {} != file dims {}",
                        query.len(),
                        r.dims()
                    )),
                );
            }
            match r.search(&query, k) {
                Ok(hits) => {
                    let hits_json = hits
                        .iter()
                        .map(|h| format!("{{\"id\":{},\"distance\":{:.6}}}", h.id, h.distance))
                        .collect::<Vec<_>>()
                        .join(",");
                    let body = format!(
                        "{{\"query\":[{}],\"k\":{k},\"hits\":[{hits_json}]}}",
                        query
                            .iter()
                            .map(|f| format!("{:.6}", f))
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                    ("200 OK".into(), body)
                }
                Err(e) => (
                    "500 Internal Server Error".into(),
                    json_error(&e.to_string()),
                ),
            }
        }
        Err(e) => ("500 Internal Server Error".into(), json_error(&e)),
    }
}

fn handle_api_validate(
    params: &HashMap<String, String>,
    default_file: Option<&Path>,
) -> (String, String) {
    let file = params
        .get("file")
        .map(|s| PathBuf::from(s))
        .or_else(|| default_file.map(|p| p.to_path_buf()));
    let p = match file {
        Some(f) => f,
        None => return ("400 Bad Request".into(), json_error("no file")),
    };
    match crate::validate::validate(&p) {
        Ok(rep) => {
            let warnings = rep
                .warnings
                .iter()
                .map(|w| format!("\"{}\"", json_escape(w)))
                .collect::<Vec<_>>()
                .join(",");
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
        Err(e) => (
            "500 Internal Server Error".into(),
            json_error(&e.to_string()),
        ),
    }
}

fn handle_api_stats(
    params: &HashMap<String, String>,
    default_file: Option<&Path>,
) -> (String, String) {
    let file = params
        .get("file")
        .map(|s| PathBuf::from(s))
        .or_else(|| default_file.map(|p| p.to_path_buf()));
    let p = match file {
        Some(f) => f,
        None => return ("400 Bad Request".into(), json_error("no file")),
    };
    match open_reader(&p) {
        Ok(r) => {
            let file_size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            let data_bytes = r.len() * r.dims() as u64;
            let original = data_bytes * 4;
            let ratio = if data_bytes == 0 {
                0.0
            } else {
                original as f64 / data_bytes as f64
            };
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

fn parse_vector(s: &str) -> Result<Vec<f32>, String> {
    if s.trim().is_empty() {
        return Err("empty vector".into());
    }
    s.split(',')
        .map(|x| {
            x.trim()
                .parse::<f32>()
                .map_err(|_| format!("bad float '{x}'"))
        })
        .collect()
}

fn handle_api_quantize(
    params: &HashMap<String, String>,
    default_file: Option<&Path>,
) -> (String, String) {
    let file = params
        .get("file")
        .map(|s| PathBuf::from(s))
        .or_else(|| default_file.map(|p| p.to_path_buf()));
    let p = match file {
        Some(f) => f,
        None => {
            return (
                "400 Bad Request".into(),
                json_error("no file for quantizer"),
            )
        }
    };
    let vec_str = match params
        .get("vector")
        .or_else(|| params.get("query"))
        .or_else(|| params.get("v"))
    {
        Some(v) => v,
        None => {
            return (
                "400 Bad Request".into(),
                json_error("missing vector (comma-separated)"),
            )
        }
    };
    let vec = match parse_vector(vec_str) {
        Ok(v) => v,
        Err(e) => return ("400 Bad Request".into(), json_error(&e)),
    };
    match open_reader(&p) {
        Ok(r) => {
            if vec.len() != r.dims() {
                return (
                    "400 Bad Request".into(),
                    json_error(&format!(
                        "vector dims {} != file dims {}",
                        vec.len(),
                        r.dims()
                    )),
                );
            }
            match r.quantizer().quantize_vector(&vec) {
                Ok(q) => {
                    let dq = match r.quantizer().dequantize_vector(&q) {
                        Ok(v) => v,
                        Err(e) => {
                            return (
                                "500 Internal Server Error".into(),
                                json_error(&e.to_string()),
                            )
                        }
                    };
                    let q_str = q
                        .iter()
                        .map(|b| b.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    let v_str = vec
                        .iter()
                        .map(|f| format!("{:.6}", f))
                        .collect::<Vec<_>>()
                        .join(",");
                    let dq_str = dq
                        .iter()
                        .map(|f| format!("{:.6}", f))
                        .collect::<Vec<_>>()
                        .join(",");
                    let err: f32 = vec
                        .iter()
                        .zip(dq.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f32::max);
                    let mse = vec
                        .iter()
                        .zip(dq.iter())
                        .map(|(a, b)| {
                            let d = a - b;
                            d * d
                        })
                        .sum::<f32>()
                        / vec.len() as f32;
                    let body = format!("{{\"vector\":[{v_str}],\"quantized\":[{q_str}],\"dequantized\":[{dq_str}],\"max_error\":{err:.6},\"mse\":{mse:.6}}}");
                    ("200 OK".into(), body)
                }
                Err(e) => (
                    "500 Internal Server Error".into(),
                    json_error(&e.to_string()),
                ),
            }
        }
        Err(e) => ("500 Internal Server Error".into(), json_error(&e)),
    }
}

fn handle_api_distance(params: &HashMap<String, String>) -> (String, String) {
    let a_str = match params.get("a") {
        Some(v) => v,
        None => return ("400 Bad Request".into(), json_error("missing a")),
    };
    let b_str = match params.get("b") {
        Some(v) => v,
        None => return ("400 Bad Request".into(), json_error("missing b")),
    };
    let metric_str = params.get("metric").map(|s| s.as_str()).unwrap_or("l2");
    let a = match parse_vector(a_str) {
        Ok(v) => v,
        Err(e) => return ("400 Bad Request".into(), json_error(&e)),
    };
    let b = match parse_vector(b_str) {
        Ok(v) => v,
        Err(e) => return ("400 Bad Request".into(), json_error(&e)),
    };
    let res: Result<f32, String> = match metric_str.to_lowercase().as_str() {
        "l2" | "euclidean" => crate::distance::l2_squared(&a, &b).map_err(|e| e.to_string()),
        "l2norm" => crate::distance::l2(&a, &b).map_err(|e| e.to_string()),
        "cos" | "cosine" => crate::distance::cosine_distance(&a, &b).map_err(|e| e.to_string()),
        "dot" => crate::distance::dot(&a, &b).map_err(|e| e.to_string()),
        "ip" => crate::distance::inner_product_distance(&a, &b).map_err(|e| e.to_string()),
        other => {
            return (
                "400 Bad Request".into(),
                json_error(&format!("unknown metric {other}")),
            )
        }
    };
    match res {
        Ok(d) => (
            "200 OK".into(),
            format!(
                "{{\"a\":[{}],\"b\":[{}],\"metric\":\"{}\",\"distance\":{:.6}}}",
                a.iter()
                    .map(|x| format!("{:.6}", x))
                    .collect::<Vec<_>>()
                    .join(","),
                b.iter()
                    .map(|x| format!("{:.6}", x))
                    .collect::<Vec<_>>()
                    .join(","),
                json_escape(metric_str),
                d
            ),
        ),
        Err(e) => ("400 Bad Request".into(), json_error(&e)),
    }
}

fn handle_api_transform(params: &HashMap<String, String>) -> (String, String) {
    let vec_str = match params.get("vector").or_else(|| params.get("v")) {
        Some(v) => v,
        None => return ("400 Bad Request".into(), json_error("missing vector")),
    };
    let vec = match parse_vector(vec_str) {
        Ok(v) => v,
        Err(e) => return ("400 Bad Request".into(), json_error(&e)),
    };
    let typ = params
        .get("type")
        .or_else(|| params.get("t"))
        .map(|s| s.as_str())
        .unwrap_or("normalize");
    let out: Result<Vec<f32>, String> = match typ.to_lowercase().as_str() {
        "normalize" | "norm" => {
            let t = crate::transform::Normalizer::new(vec.len());
            let mut o = vec![0.0; vec.len()];
            if let Err(e) = t.transform(&vec, &mut o) {
                Err(e.to_string())
            } else {
                Ok(o)
            }
        }
        "identity" | "none" => Ok(vec.clone()),
        _ => Err(format!("unknown transform {typ} (normalize, identity)")),
    };
    match out {
        Ok(o) => {
            let in_str = vec
                .iter()
                .map(|x| format!("{:.6}", x))
                .collect::<Vec<_>>()
                .join(",");
            let out_str = o
                .iter()
                .map(|x| format!("{:.6}", x))
                .collect::<Vec<_>>()
                .join(",");
            (
                "200 OK".into(),
                format!(
                    "{{\"input\":[{in_str}],\"type\":\"{}\",\"output\":[{out_str}]}}",
                    json_escape(typ)
                ),
            )
        }
        Err(e) => ("400 Bad Request".into(), json_error(&e)),
    }
}

fn handle_api_dataset(params: &HashMap<String, String>) -> (String, String) {
    let dims: usize = params.get("dims").and_then(|s| s.parse().ok()).unwrap_or(4);
    let count: usize = params
        .get("count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    if dims == 0 || dims > 1024 {
        return ("400 Bad Request".into(), json_error("dims 1..1024"));
    }
    if count == 0 || count > 100 {
        return ("400 Bad Request".into(), json_error("count 1..100"));
    }
    let seed: u64 = params
        .get("seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let typ = params.get("type").map(|s| s.as_str()).unwrap_or("uniform");
    let ds = match typ {
        "uniform" => crate::dataset::Dataset::synthetic_uniform(count, dims, -1.0, 1.0, seed),
        "clustered" => crate::dataset::Dataset::synthetic_clustered(count, dims, 3, 0.2, seed),
        _ => {
            return (
                "400 Bad Request".into(),
                json_error("type uniform or clustered"),
            )
        }
    };
    let vecs_json = ds
        .vectors
        .iter()
        .map(|v| {
            format!(
                "[{}]",
                v.iter()
                    .map(|x| format!("{:.4}", x))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    ("200 OK".into(), format!("{{\"dims\":{dims},\"count\":{count},\"type\":\"{}\",\"seed\":{seed},\"vectors\":[{vecs_json}]}}", json_escape(typ)))
}

fn handle_api_batch(
    params: &HashMap<String, String>,
    default_file: Option<&Path>,
) -> (String, String) {
    let file = params
        .get("file")
        .map(|s| PathBuf::from(s))
        .or_else(|| default_file.map(|p| p.to_path_buf()));
    let p = match file {
        Some(f) => f,
        None => return ("400 Bad Request".into(), json_error("no file")),
    };
    let start: u64 = params
        .get("start")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let count: u64 = params
        .get("count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    if count > 100 {
        return ("400 Bad Request".into(), json_error("count max 100"));
    }
    match open_reader(&p) {
        Ok(r) => {
            let mut out = Vec::new();
            for i in 0..count {
                let id = start + i;
                if id >= r.len() {
                    break;
                }
                match r.get_vector(id) {
                    Ok(v) => {
                        let s = v
                            .iter()
                            .map(|x| format!("{:.4}", x))
                            .collect::<Vec<_>>()
                            .join(",");
                        out.push(format!("{{\"id\":{id},\"vector\":[{s}]}}"));
                    }
                    Err(e) => {
                        return (
                            "500 Internal Server Error".into(),
                            json_error(&e.to_string()),
                        )
                    }
                }
            }
            (
                "200 OK".into(),
                format!(
                    "{{\"start\":{start},\"count\":{},\"vectors\":[{}]}}",
                    out.len(),
                    out.join(",")
                ),
            )
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
        write_json(
            &mut stream,
            "405 Method Not Allowed",
            &json_error("only GET"),
        );
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
            write_response(
                &mut stream,
                "200 OK",
                "text/html; charset=utf-8",
                GUI_HTML.as_bytes(),
            );
        }
        "/style.css" => {
            write_response(
                &mut stream,
                "200 OK",
                "text/css; charset=utf-8",
                GUI_CSS.as_bytes(),
            );
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
        "/api/quantize" => {
            let (status, body) = handle_api_quantize(&params, default_ref);
            write_json(&mut stream, &status, &body);
        }
        "/api/distance" => {
            let (status, body) = handle_api_distance(&params);
            write_json(&mut stream, &status, &body);
        }
        "/api/transform" => {
            let (status, body) = handle_api_transform(&params);
            write_json(&mut stream, &status, &body);
        }
        "/api/dataset" => {
            let (status, body) = handle_api_dataset(&params);
            write_json(&mut stream, &status, &body);
        }
        "/api/batch" => {
            let (status, body) = handle_api_batch(&params, default_ref);
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

const GUI_CSS: &str = r##"
:root{--bg:#0a0a0b;--fg:#e8e8e8;--muted:#9aa0a6;--accent:#6ee7b7;--accent2:#38bdf8;--card:#141416;--border:#1f1f23;--code:#151518;--hover:#1d1d20}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
a{color:var(--accent2)}header{padding:16px 20px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;background:rgba(10,10,11,0.85);backdrop-filter:blur(8px);z-index:20}
header h1{margin:0;font-size:18px;letter-spacing:-0.02em}header h1 span{color:var(--accent)}
.layout{display:flex;min-height:calc(100vh - 57px)}
.sidebar{width:200px;border-right:1px solid var(--border);background:var(--card);padding:12px;display:flex;flex-direction:column;gap:6px;position:sticky;top:57px;height:calc(100vh - 57px);overflow:auto}
.sidebar button{text-align:left;padding:10px 12px;border-radius:8px;border:1px solid transparent;background:transparent;color:var(--muted);cursor:pointer;font:13px ui-sans-serif}
.sidebar button.active,.sidebar button:hover{background:var(--code);color:var(--fg);border-color:var(--border)}
.sidebar .hint{font:11px ui-monospace,monospace;color:var(--muted);padding:8px 4px}
.main{flex:1;padding:20px;max-width:1100px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.layout{flex-direction:column}.sidebar{width:100%;height:auto;position:static;flex-direction:row;overflow:auto}.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
.card h3{margin:0 0 10px;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted)}
input,select,button,textarea{font:13px ui-sans-serif;padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:var(--code);color:var(--fg)}
textarea{width:100%;resize:vertical;min-height:60px}
button{background:var(--fg);color:var(--bg);font-weight:600;cursor:pointer}
button.ghost{background:transparent;color:var(--fg)}
button:hover{opacity:0.9}
input:focus,select:focus,textarea:focus{outline:none;border-color:var(--accent2)}
pre{background:var(--code);border:1px solid var(--border);border-radius:8px;padding:12px;overflow:auto;font:12px ui-monospace,monospace;max-height:320px;white-space:pre-wrap;word-break:break-word}
.badge{font:11px ui-monospace,monospace;padding:4px 8px;border:1px solid var(--border);border-radius:999px;background:var(--card);color:var(--muted)}
canvas{width:100%;height:140px;background:var(--code);border:1px solid var(--border);border-radius:8px}
.kv{display:grid;grid-template-columns:140px 1fr;gap:6px;font-size:13px}.kv div:nth-child(odd){color:var(--muted)}
.tab{display:none}.tab.active{display:block}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.table{width:100%;border-collapse:collapse;font-size:13px}
.table th, .table td{padding:8px 10px;border-bottom:1px solid var(--border);text-align:left}
.table th{color:var(--muted);font-weight:600;font-size:11px;letter-spacing:0.06em;text-transform:uppercase}
.pill{font:11px ui-monospace,monospace;padding:2px 6px;border-radius:999px;background:var(--code);border:1px solid var(--border)}
.toast{position:fixed;bottom:16px;right:16px;background:var(--fg);color:var(--bg);padding:10px 14px;border-radius:8px;font:13px ui-sans-serif;box-shadow:0 8px 24px rgba(0,0,0,0.4);display:none}
"##;

const GUI_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>bitcompact — serve GUI</title>
<link rel="stylesheet" href="/style.css">
</head>
<body>
<header>
  <div>
    <h1>bit<span>-compact</span> <small style="font-weight:400;color:var(--muted);margin-left:8px;">serve</small></h1>
    <div style="font:12px ui-monospace,monospace;color:var(--muted);margin-top:2px;" id="headerFile">no file</div>
  </div>
  <div style="display:flex;gap:8px;align-items:center;">
    <input id="fileInput" placeholder="/tmp/vectors.btcp" style="width:260px">
    <button onclick="loadInfo()">Open</button>
    <span class="badge" id="headerBadge">offline</span>
  </div>
</header>

<div class="layout">
  <nav class="sidebar">
    <button data-tab="dashboard" class="active">Dashboard</button>
    <button data-tab="vectors">Vectors</button>
    <button data-tab="search">Search</button>
    <button data-tab="quantize">Quantize</button>
    <button data-tab="distance">Distance</button>
    <button data-tab="transform">Transform</button>
    <button data-tab="dataset">Dataset</button>
    <button data-tab="validate">Validate</button>
    <button data-tab="create">Create</button>
    <div class="hint">1 seek · 0 alloc<br>BTCP 32B · SHA-256<br><span id="sidebarStats">—</span></div>
  </nav>

  <main class="main">
    <!-- Dashboard -->
    <section id="tab-dashboard" class="tab active">
      <h2 style="margin:0 0 12px;">Dashboard</h2>
      <div class="grid">
        <div class="card"><h3>File info</h3><div class="kv" id="info">loading…</div><div class="row" style="margin-top:12px"><button onclick="loadInfo()">Reload</button><button class="ghost" onclick="doValidate()">Validate</button></div><pre id="validateOut" style="display:none;margin-top:10px"></pre></div>
        <div class="card"><h3>Stats</h3><div class="kv" id="stats">—</div><button class="ghost" onclick="loadStats()" style="margin-top:12px">Refresh</button></div>
      </div>
      <div class="card" style="margin-top:14px"><h3>Quick actions</h3><div class="row"><button onclick="switchTab('vectors')">Browse vectors</button><button onclick="switchTab('search')">Search</button><button onclick="switchTab('quantize')">Quantize playground</button><button class="ghost" onclick="loadBatch()">Load batch 0-9</button></div><pre id="quickLog">GUI ready. All APIs: /api/info, /api/get, /api/batch, /api/search, /api/quantize, /api/distance, /api/transform, /api/dataset, /api/validate, /api/stats</pre></div>
    </section>

    <!-- Vectors -->
    <section id="tab-vectors" class="tab">
      <h2 style="margin:0 0 12px;">Vectors</h2>
      <div class="card">
        <div class="row"><input id="idInput" type="number" min="0" value="0" style="width:90px"><button onclick="loadVector()">Get</button><button class="ghost" onclick="prevVec()">◀</button><button class="ghost" onclick="nextVec()">▶</button><span class="pill" id="vecCount">count ?</span><input id="batchStart" type="number" value="0" style="width:80px" placeholder="start"><input id="batchCount" type="number" value="5" style="width:60px"><button class="ghost" onclick="loadBatch()">Batch</button></div>
        <canvas id="chart" width="700" height="140"></canvas>
        <pre id="vecOut">—</pre>
        <div id="batchOut"></div>
      </div>
    </section>

    <!-- Search -->
    <section id="tab-search" class="tab">
      <h2 style="margin:0 0 12px;">Search</h2>
      <div class="card">
        <label style="font:12px ui-monospace,monospace;color:var(--muted)">Query (comma-separated, dims auto from file)</label>
        <textarea id="queryInput" placeholder="0.1,0.2,0.3,..."></textarea>
        <div class="row" style="margin-top:8px"><select id="metricInput"><option value="l2">L2 squared</option><option value="cosine">Cosine</option><option value="dot">Dot</option><option value="ip">Inner prod</option></select><input id="kInput" type="number" min="1" max="100" value="5" style="width:80px"><button onclick="doSearch()">Search</button><button class="ghost" onclick="fillRandomQuery()">Random</button></div>
        <pre id="searchOut">—</pre>
        <canvas id="searchChart" width="700" height="120"></canvas>
      </div>
    </section>

    <!-- Quantize -->
    <section id="tab-quantize" class="tab">
      <h2 style="margin:0 0 12px;">Quantize playground</h2>
      <div class="card">
        <label style="font:12px ui-monospace,monospace;color:var(--muted)">Input vector (comma-separated)</label>
        <textarea id="qInput" placeholder="0.5,1.0,2.0,..."></textarea>
        <div class="row" style="margin-top:8px"><button onclick="doQuantize()">Quantize → Dequantize</button><button class="ghost" onclick="fillVectorFromCurrent()">Use current vector</button></div>
        <canvas id="quantChart" width="700" height="140"></canvas>
        <pre id="quantOut">—</pre>
      </div>
    </section>

    <!-- Distance -->
    <section id="tab-distance" class="tab">
      <h2 style="margin:0 0 12px;">Distance</h2>
      <div class="grid">
        <div class="card"><h3>Vector A</h3><textarea id="distA" placeholder="0,1,2,3"></textarea></div>
        <div class="card"><h3>Vector B</h3><textarea id="distB" placeholder="3,2,1,0"></textarea></div>
      </div>
      <div class="row" style="margin-top:8px"><select id="distMetric"><option value="l2">L2 squared</option><option value="l2norm">L2</option><option value="cosine">Cosine</option><option value="dot">Dot</option><option value="ip">Inner</option></select><button onclick="doDistance()">Compute</button></div>
      <pre id="distOut">—</pre>
    </section>

    <!-- Transform -->
    <section id="tab-transform" class="tab">
      <h2 style="margin:0 0 12px;">Transform</h2>
      <div class="card">
        <textarea id="transInput" placeholder="3,4,..."></textarea>
        <div class="row" style="margin-top:8px"><select id="transType"><option value="normalize">Normalize (unit)</option><option value="identity">Identity</option></select><button onclick="doTransform()">Apply</button></div>
        <canvas id="transChart" width="700" height="140"></canvas>
        <pre id="transOut">—</pre>
      </div>
    </section>

    <!-- Dataset -->
    <section id="tab-dataset" class="tab">
      <h2 style="margin:0 0 12px;">Dataset generator</h2>
      <div class="card">
        <div class="row"><label>dims <input id="dsDims" type="number" value="4" style="width:70px"></label><label>count <input id="dsCount" type="number" value="5" style="width:70px"></label><label>seed <input id="dsSeed" type="number" value="42" style="width:70px"></label><select id="dsType"><option value="uniform">Uniform</option><option value="clustered">Clustered</option></select><button onclick="doDataset()">Generate</button></div>
        <pre id="dsOut">—</pre>
      </div>
    </section>

    <!-- Validate -->
    <section id="tab-validate" class="tab">
      <h2 style="margin:0 0 12px;">Validate</h2>
      <div class="card"><button onclick="doValidateFull()">Run full validation</button><pre id="validateFull" style="margin-top:10px">—</pre></div>
    </section>

    <!-- Create -->
    <section id="tab-create" class="tab">
      <h2 style="margin:0 0 12px;">Create</h2>
      <div class="card"><p style="color:var(--muted);font-size:13px">Create a new .btcp file from pasted JSON array. Uses <code>Quantizer::calibrate</code> then <code>CompactWriter</code>. This is a dry-run preview — actual file creation via CLI <code>bitcompact create</code> is recommended for large data.</p>
      <textarea id="createInput" placeholder='[[0,1,2],[3,4,5]]' style="min-height:100px"></textarea>
      <div class="row" style="margin-top:8px"><input id="createPath" placeholder="/tmp/new.btcp" style="flex:1"><button onclick="alert('Use CLI: bitcompact create '+document.getElementById('createPath').value)">Preview</button></div><pre id="createOut">—</pre></div>
    </section>
  </main>
</div>

<div class="toast" id="toast"></div>

<script>
let dims=0, count=0, currentId=0, currentVector=null;
function fileParam(){ const v=document.getElementById('fileInput').value.trim(); return v? `&file=${encodeURIComponent(v)}` : ''; }
function fileQuery(){ const v=document.getElementById('fileInput').value.trim(); return v? `?file=${encodeURIComponent(v)}` : ''; }
function toast(m){ const t=document.getElementById('toast'); t.textContent=m; t.style.display='block'; setTimeout(()=>t.style.display='none',2000); }
async function fetchJSON(url){ const r=await fetch(url); const t=await r.text(); try{ return JSON.parse(t);}catch{ return {error:t, status:r.status}; } }
function setHeaderFile(p){ document.getElementById('headerFile').textContent=p||'no file'; }

async function loadInfo(){
  const q=fileQuery();
  const j=await fetchJSON('/api/info'+q);
  const el=document.getElementById('info');
  if(j.error){ el.innerHTML=`<div style="color:#f87171">${j.error}</div>`; document.getElementById('headerBadge').textContent='error'; return; }
  dims=j.dims; count=j.count; setHeaderFile(j.path); document.getElementById('headerBadge').textContent=`${j.dims}D · ${j.count}`;
  document.getElementById('vecCount').textContent=`count ${count}`;
  document.getElementById('sidebarStats').textContent=`${dims}D · ${count} vecs`;
  el.innerHTML=`<div>path</div><div style="word-break:break-all">${j.path}</div><div>dims</div><div>${j.dims}</div><div>count</div><div>${j.count}</div><div>quant</div><div>${j.quant}</div><div>distance</div><div>${j.distance}</div><div>footer</div><div>${j.footer_offset}</div><div>checksum</div><div>${j.checksum}</div><div>size</div><div>${j.file_size} B</div>`;
  document.getElementById('queryInput').placeholder=Array(dims).fill('0.0').join(',');
  document.getElementById('qInput').placeholder=Array(dims).fill('0.5').join(',');
  document.getElementById('distA').placeholder=Array(dims).fill('0').join(',');
  document.getElementById('distB').placeholder=Array(dims).fill('1').join(',');
  loadStats();
}
async function loadStats(){
  const q=fileQuery();
  const j=await fetchJSON('/api/stats'+q);
  const el=document.getElementById('stats');
  if(j.error){ el.textContent=j.error; return; }
  el.innerHTML=`<div>file</div><div style="word-break:break-all">${j.path}</div><div>dims × count</div><div>${j.dims} × ${j.count}</div><div>data</div><div>${j.data_bytes} B</div><div>original</div><div>${j.original_bytes} B</div><div>ratio</div><div>${j.ratio}×</div><div>file size</div><div>${j.file_size} B</div>`;
}
async function loadVector(){
  const id=parseInt(document.getElementById('idInput').value||'0',10);
  currentId=isNaN(id)?0:id;
  if(currentId<0) currentId=0; if(currentId>=count) currentId=count-1;
  document.getElementById('idInput').value=currentId;
  const j=await fetchJSON(`/api/get?id=${currentId}${fileParam()}`);
  const el=document.getElementById('vecOut');
  if(j.error){ el.textContent=j.error; return; }
  currentVector=j.vector;
  el.textContent=`id ${j.id}\nquantized [${j.quantized.join(',')}]\nvector [${j.vector.map(v=>Number(v).toFixed(4)).join(', ')}]`;
  drawChart('chart', j.vector);
  toast(`get id=${j.id}`);
}
function prevVec(){ let v=parseInt(document.getElementById('idInput').value||'0',10)-1; document.getElementById('idInput').value=v; loadVector(); }
function nextVec(){ let v=parseInt(document.getElementById('idInput').value||'0',10)+1; document.getElementById('idInput').value=v; loadVector(); }
async function loadBatch(){
  const s=parseInt(document.getElementById('batchStart').value||'0',10);
  const c=parseInt(document.getElementById('batchCount').value||'5',10);
  const j=await fetchJSON(`/api/batch?start=${s}&count=${c}${fileParam()}`);
  const el=document.getElementById('batchOut');
  if(j.error){ el.textContent=j.error; return; }
  let html='<table class="table"><tr><th>id</th><th>vector (first 6)</th></tr>';
  j.vectors.forEach(v=>{ html+=`<tr><td>${v.id}</td><td>${v.vector.slice(0,6).map(x=>Number(x).toFixed(2)).join(', ')}${v.vector.length>6?' …':''}</td></tr>`; });
  html+='</table>';
  el.innerHTML=html;
}
function fillRandomQuery(){
  if(!dims) return;
  const q=Array.from({length:dims},()=> (Math.random()*2-1).toFixed(3)).join(',');
  document.getElementById('queryInput').value=q;
}
async function doSearch(){
  const q=document.getElementById('queryInput').value.trim();
  const k=document.getElementById('kInput').value||'5';
  const metric=document.getElementById('metricInput').value;
  // we currently send to /api/search which uses file's metric, metric selector is for future; we map to query search with metric via distance endpoint? For now use search with file metric
  if(!q){ toast('enter query'); return; }
  const j=await fetchJSON(`/api/search?query=${encodeURIComponent(q)}&k=${k}${fileParam()}`);
  const el=document.getElementById('searchOut');
  if(j.error){ el.textContent=j.error; return; }
  el.textContent=j.hits.map(h=>`id=${h.id} dist=${Number(h.distance).toFixed(6)}`).join('\n') || 'no hits';
  drawBar('searchChart', j.hits.map(h=>h.distance), j.hits.map(h=>'#'+h.id));
  toast(`search k=${j.k} hits=${j.hits.length}`);
}
async function doQuantize(){
  const v=document.getElementById('qInput').value.trim();
  if(!v){ toast('enter vector'); return; }
  const j=await fetchJSON(`/api/quantize?vector=${encodeURIComponent(v)}${fileParam()}`);
  const el=document.getElementById('quantOut');
  if(j.error){ el.textContent=j.error; return; }
  el.textContent=`quantized [${j.quantized.join(',')}]\ndequant [${j.dequantized.map(x=>Number(x).toFixed(4)).join(',')}]\nmax_error ${j.max_error} mse ${j.mse}`;
  // overlay chart
  drawQuantChart(j.vector, j.dequantized);
}
function fillVectorFromCurrent(){
  if(currentVector) document.getElementById('qInput').value=currentVector.join(',');
}
async function doDistance(){
  const a=document.getElementById('distA').value.trim();
  const b=document.getElementById('distB').value.trim();
  const m=document.getElementById('distMetric').value;
  const j=await fetchJSON(`/api/distance?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}&metric=${m}`);
  document.getElementById('distOut').textContent=j.error? j.error : `metric ${j.metric} distance ${Number(j.distance).toFixed(6)}\na [${j.a.slice(0,6).join(',')}]\nb [${j.b.slice(0,6).join(',')}]`;
}
async function doTransform(){
  const v=document.getElementById('transInput').value.trim();
  const t=document.getElementById('transType').value;
  const j=await fetchJSON(`/api/transform?vector=${encodeURIComponent(v)}&type=${t}`);
  const el=document.getElementById('transOut');
  if(j.error){ el.textContent=j.error; return; }
  el.textContent=`input [${j.input.slice(0,8).join(',')}]\n${j.type} -> [${j.output.slice(0,8).join(',')}]`;
  drawChart('transChart', j.output);
}
async function doDataset(){
  const dims=document.getElementById('dsDims').value||'4';
  const count=document.getElementById('dsCount').value||'5';
  const seed=document.getElementById('dsSeed').value||'42';
  const type=document.getElementById('dsType').value;
  const j=await fetchJSON(`/api/dataset?dims=${dims}&count=${count}&seed=${seed}&type=${type}`);
  const el=document.getElementById('dsOut');
  if(j.error){ el.textContent=j.error; return; }
  el.textContent=j.vectors.map((v,i)=>`[${i}] ${v.slice(0,6).join(', ')}${v.length>6?' …':''}`).join('\n');
}
async function doValidate(){ const j=await fetchJSON('/api/validate'+fileQuery()); document.getElementById('validateOut').style.display='block'; document.getElementById('validateOut').textContent=JSON.stringify(j,null,2); }
async function doValidateFull(){ const j=await fetchJSON('/api/validate'+fileQuery()); document.getElementById('validateFull').textContent=JSON.stringify(j,null,2); }
function drawChart(id, vec){
  const c=document.getElementById(id);
  if(!c) return;
  const ctx=c.getContext('2d');
  const w=c.width, h=c.height;
  ctx.clearRect(0,0,w,h);
  if(!vec||!vec.length) return;
  const min=Math.min(...vec), max=Math.max(...vec), range=(max-min)||1;
  ctx.strokeStyle='#6ee7b7'; ctx.lineWidth=1.5; ctx.beginPath();
  vec.forEach((v,i)=>{ const x=(i/(vec.length-1))*w; const y=h-((v-min)/range)*h*0.9 - h*0.05; if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
  ctx.stroke();
  ctx.fillStyle='#38bdf8';
  vec.forEach((v,i)=>{ const x=(i/(vec.length-1))*w; const y=h-((v-min)/range)*h*0.9 - h*0.05; ctx.beginPath(); ctx.arc(x,y,2,0,Math.PI*2); ctx.fill(); });
}
function drawQuantChart(orig, deq){
  const c=document.getElementById('quantChart');
  const ctx=c.getContext('2d');
  const w=c.width, h=c.height;
  ctx.clearRect(0,0,w,h);
  if(!orig||!deq) return;
  const all=orig.concat(deq);
  const min=Math.min(...all), max=Math.max(...all), range=(max-min)||1;
  const draw=(vec,color)=>{
    ctx.strokeStyle=color; ctx.lineWidth=1.5; ctx.beginPath();
    vec.forEach((v,i)=>{ const x=(i/(vec.length-1))*w; const y=h-((v-min)/range)*h*0.9 - h*0.05; if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
    ctx.stroke();
  };
  draw(orig,'#6ee7b7'); draw(deq,'#38bdf8');
}
function drawBar(id, vals, labels){
  const c=document.getElementById(id);
  if(!c) return;
  const ctx=c.getContext('2d');
  const w=c.width, h=c.height;
  ctx.clearRect(0,0,w,h);
  if(!vals.length) return;
  const max=Math.max(...vals);
  const barW=w/vals.length*0.7;
  const gap=w/vals.length*0.3;
  vals.forEach((v,i)=>{
    const bh=(v/max)*h*0.8;
    const x=i*(barW+gap)+gap/2;
    const y=h-bh;
    ctx.fillStyle='#6ee7b7';
    ctx.fillRect(x,y,barW,bh);
    ctx.fillStyle='#9aa0a6';
    ctx.font='10px ui-monospace,monospace';
    ctx.fillText(labels[i], x, h-2);
  });
}
function switchTab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.sidebar button').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  document.querySelector(`.sidebar button[data-tab="${name}"]`).classList.add('active');
  if(name==='vectors' && dims) loadVector();
}
document.querySelectorAll('.sidebar button').forEach(b=>b.addEventListener('click',()=>switchTab(b.dataset.tab)));
loadInfo();
</script>
</body>
</html>
"##;
