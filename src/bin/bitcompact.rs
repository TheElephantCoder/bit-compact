use std::env;
use std::fs;
use std::path::Path;

use bit_compact::{CompactReader, CompactWriter, DistanceMetric, QuantType, Quantizer, validate};

fn print_usage() {
    eprintln!(
        r#"bitcompact — bit-compact CLI (zero deps)

Usage:
  bitcompact create <file> [--dims N] [--metric L2|cosine|ip] [--align]
  bitcompact info   <file>
  bitcompact get    <file> <id>
  bitcompact search <file> --query x,y,z --k K
  bitcompact validate <file>
  bitcompact stats  <file>
  bitcompact serve [file] [--port PORT] [--host HOST]

Examples:
  bitcompact create vectors.btcp --dims 128 --metric cosine --align
  bitcompact info vectors.btcp
  bitcompact get vectors.btcp 42
  bitcompact search vectors.btcp --query 0.1,0.2,0.3 --k 5
  bitcompact validate vectors.btcp
  bitcompact serve vectors.btcp --port 8080
  bitcompact serve --port 3000
"#
    );
}

fn parse_metric(s: &str) -> Result<DistanceMetric, String> {
    match s.to_lowercase().as_str() {
        "l2" | "euclidean" => Ok(DistanceMetric::L2),
        "cosine" | "cos" => Ok(DistanceMetric::Cosine),
        "ip" | "inner" | "innerproduct" => Ok(DistanceMetric::InnerProduct),
        other => Err(format!("unknown metric {other} (L2|cosine|ip)")),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }
    let res: Result<(), String> = match args[1].as_str() {
        "create" => cmd_create(&args[2..]),
        "info" => cmd_info(&args[2..]),
        "get" => cmd_get(&args[2..]),
        "search" => cmd_search(&args[2..]),
        "validate" => cmd_validate(&args[2..]),
        "stats" => cmd_stats(&args[2..]),
        "serve" => cmd_serve(&args[2..]),
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        other => Err(format!("unknown command {other}")),
    };
    if let Err(e) = res {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn cmd_create(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("create <file> [--dims N] [--metric M] [--align]".into());
    }
    let file = &args[0];
    let mut dims: Option<usize> = None;
    let mut metric = DistanceMetric::L2;
    let mut align = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dims" => {
                dims = Some(args.get(i + 1).ok_or("--dims needs value")?.parse::<usize>().map_err(|e| e.to_string())?);
                i += 2;
            }
            "--metric" => {
                metric = parse_metric(args.get(i + 1).ok_or("--metric needs value")?)?;
                i += 2;
            }
            "--align" => { align = true; i += 1; }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    // For demo, create empty file with dummy quantizer (min 0 max 1 per dim)
    let d = dims.unwrap_or(8);
    if d == 0 || d > 65535 { return Err("dims must be 1..65535".into()); }
    let mins = vec![0.0f32; d];
    let maxs = vec![1.0f32; d];
    let q = Quantizer::new(mins, maxs).map_err(|e| e.to_string())?;
    let w = CompactWriter::create_with_version(Path::new(file), q, QuantType::SQ8, metric, 1, 0)
        .map_err(|e| e.to_string())?;
    // Write no vectors, just finalize (creates valid empty file)
    if align {
        w.finalize_with_padding(true).map_err(|e| e.to_string())?;
    } else {
        w.finalize().map_err(|e| e.to_string())?;
    }
    println!("created {file} dims={d} metric={:?} align={align}", metric);
    Ok(())
}

fn cmd_info(args: &[String]) -> Result<(), String> {
    if args.len() != 1 { return Err("info <file>".into()); }
    let path = &args[0];
    let r = CompactReader::open(path).map_err(|e| e.to_string())?;
    println!("file: {path}");
    println!("  dims: {}", r.dims());
    println!("  count: {}", r.len());
    println!("  quant: {:?}", r.quant_type());
    println!("  distance: {:?}", r.distance_metric());
    println!("  footer_offset: {}", r.footer_offset());
    println!("  header: {:?}", r.header());
    Ok(())
}

fn cmd_get(args: &[String]) -> Result<(), String> {
    if args.len() != 2 { return Err("get <file> <id>".into()); }
    let r = CompactReader::open(&args[0]).map_err(|e| e.to_string())?;
    let id: u64 = args[1].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
    let v = r.get_vector(id).map_err(|e| e.to_string())?;
    println!("id {id}: {:?}", v);
    // also show quantized
    let q = r.get_quantized(id).map_err(|e| e.to_string())?;
    println!("quantized: {:?}", q);
    Ok(())
}

fn cmd_search(args: &[String]) -> Result<(), String> {
    if args.len() < 3 { return Err("search <file> --query x,y,z --k K".into()); }
    let file = &args[0];
    let mut query: Option<Vec<f32>> = None;
    let mut k = 5usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--query" => {
                let s = args.get(i + 1).ok_or("--query needs value")?;
                let vals: Result<Vec<f32>, _> = s.split(',').map(|x| x.trim().parse::<f32>()).collect();
                query = Some(vals.map_err(|e| e.to_string())?);
                i += 2;
            }
            "--k" => {
                k = args.get(i + 1).ok_or("--k needs value")?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
                i += 2;
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let q = query.ok_or("--query required")?;
    let r = CompactReader::open(file).map_err(|e| e.to_string())?;
    if q.len() != r.dims() { return Err(format!("query dims {} != file dims {}", q.len(), r.dims())); }
    let hits = r.search(&q, k).map_err(|e| e.to_string())?;
    println!("top {k} for query {:?}:", q);
    for h in hits { println!("  id={} dist={:.6}", h.id, h.distance); }
    Ok(())
}

fn cmd_validate(args: &[String]) -> Result<(), String> {
    if args.len() != 1 { return Err("validate <file>".into()); }
    let rep = validate::validate(&args[0]).map_err(|e| e.to_string())?;
    println!("{}", rep.summary());
    if !rep.warnings.is_empty() {
        for w in &rep.warnings { println!("  warn: {w}"); }
    }
    if rep.is_valid() {
        println!("valid: true");
        Ok(())
    } else {
        Err("file validation failed".into())
    }
}

fn cmd_stats(args: &[String]) -> Result<(), String> {
    if args.len() != 1 { return Err("stats <file>".into()); }
    let path = &args[0];
    let r = CompactReader::open(path).map_err(|e| e.to_string())?;
    let file_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let data_bytes = r.len() * r.dims() as u64;
    let original_bytes = data_bytes * 4;
    let ratio = if data_bytes == 0 { 0.0 } else { original_bytes as f64 / data_bytes as f64 };
    println!("file: {path}");
    println!("  file_size: {file_size} B");
    println!("  data_bytes (quantized): {data_bytes} B");
    println!("  original_bytes (f32): {original_bytes} B");
    println!("  ratio: {ratio:.2}x");
    println!("  dims: {} count: {}", r.dims(), r.len());
    // Sample quantization error for first vector if exists
    if r.len() > 0 {
        let qv = r.get_quantized(0).map_err(|e| e.to_string())?;
        let dq = r.quantizer().dequantize_vector(&qv).map_err(|e| e.to_string())?;
        println!("  sample id 0 quantized {:?} -> dequant {:?}", &qv[..qv.len().min(8)], &dq[..dq.len().min(4)]);
    }
    Ok(())
}

fn cmd_serve(args: &[String]) -> Result<(), String> {
    let mut file: Option<String> = None;
    let mut port: u16 = 8080;
    let mut host = "127.0.0.1".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                let p = args.get(i + 1).ok_or("--port needs value")?;
                port = p.parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
                i += 2;
            }
            "--host" => {
                host = args.get(i + 1).ok_or("--host needs value")?.clone();
                i += 2;
            }
            s if s.starts_with("--") => return Err(format!("unknown flag {s}")),
            other => {
                if file.is_some() { return Err(format!("extra arg {other}")); }
                file = Some(other.to_string());
                i += 1;
            }
        }
    }
    let path = file.map(|f| std::path::PathBuf::from(f));
    if let Some(p) = &path {
        if !p.exists() {
            // allow serving without existing file — GUI will show error until file created
            eprintln!("warning: file {} does not exist yet", p.display());
        }
    }
    bit_compact::server::serve(path, &host, port).map_err(|e| e.to_string())?;
    Ok(())
}
