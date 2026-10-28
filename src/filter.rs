//! Speakability filter: turns agent output (markdown, code, tool payloads)
//! into something reasonable to read aloud. Purely heuristic, intentionally
//! conservative — it can be disabled per connection or per `speak` message.

/// Make text "speakable": remove code blocks, URLs, markdown syntax and
/// tool-output-looking lines. Returns the cleaned text (may be empty).
pub fn make_speakable(text: &str) -> String {
    let mut out_lines: Vec<String> = Vec::new();
    let mut in_code_block = false;
    let mut code_dropped = false;

    for line in text.lines() {
        let trimmed = line.trim();

        // fenced code blocks
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            code_dropped = true;
            continue;
        }
        if in_code_block {
            continue;
        }

        if is_noise_line(trimmed) {
            continue;
        }

        out_lines.push(strip_inline_markdown(line));
    }

    let mut out = out_lines.join(" ");
    // collapse whitespace
    out = out.split_whitespace().collect::<Vec<_>>().join(" ");
    if out.is_empty() && code_dropped {
        return "Code output omitted.".to_string();
    }
    out
}

/// Lines that look like tool payloads, JSON, tables, diff markers, etc.
fn is_noise_line(line: &str) -> bool {
    if line.is_empty() {
        return true;
    }
    // JSON-ish or bracket-heavy
    if line.starts_with(['{', '[', ']', '}', '|', '<', '>']) {
        return true;
    }
    // diff / log markers
    if line.starts_with("@@ ") || (line.starts_with("--- ") && line.ends_with(" ---")) {
        return true;
    }
    // mostly punctuation/symbols -> table rows, separators, base64 blobs
    let alnum = line.chars().filter(|c| c.is_alphanumeric()).count();
    let total = line.chars().count();
    if total > 0 && (alnum as f32 / total as f32) < 0.5 {
        return true;
    }
    // JSON-ish "quoted key" lines
    if line.contains("\":") {
        return true;
    }
    false
}

/// Strip inline markdown from a single line, keeping human words.
fn strip_inline_markdown(line: &str) -> String {
    let mut s = line.to_string();

    // inline code spans: keep the content
    while let Some(start) = s.find('`') {
        if let Some(len) = s[start + 1..].find('`') {
            let inner = &s[start + 1..start + 1 + len];
            s = format!("{}{}{}", &s[..start], inner, &s[start + 1 + len + 1..]);
        } else {
            s.remove(start);
        }
    }

    // images first: ![alt](url) -> drop entirely
    s = remove_links(&s, true);
    // links: [text](url) -> text
    s = remove_links(&s, false);
    // URLs
    s = s
        .split(' ')
        .filter(|w| !w.starts_with("http://") && !w.starts_with("https://"))
        .collect::<Vec<_>>()
        .join(" ");

    // heading / list / emphasis markers
    let mut cleaned = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '#' | '*' | '_' | '~' | '>' => continue,
            _ => cleaned.push(c),
        }
    }
    cleaned.trim().to_string()
}

/// Remove markdown links/images. If `images_only`, only `![...](...)`.
fn remove_links(s: &str, images_only: bool) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        let pattern = if images_only { "](![" } else { "](http" };
        // find "](...)" pattern; walk back to matching '['
        let Some(close) = find_sub(rest, pattern) else {
            out.push_str(rest);
            break;
        };
        // start of "]("
        let bracket_pos = if images_only { close + 1 } else { close };
        let after = bracket_pos + 1; // index of ']'
        // find matching opening '[' before `after`
        let Some(open) = rest[..after].rfind('[') else {
            out.push_str(&rest[..=close]);
            rest = &rest[close + pattern.len()..];
            continue;
        };
        // find closing ')' after bracket_pos
        let Some(paren_close) = rest[bracket_pos..].find(')') else {
            out.push_str(&rest[..=close]);
            rest = &rest[close + pattern.len()..];
            continue;
        };
        let url_end = bracket_pos + paren_close;
        let text = &rest[open + 1..after - 1];
        if images_only {
            out.push_str(&rest[..open]);
        } else {
            out.push_str(&rest[..open]);
            out.push_str(text);
        }
        rest = &rest[url_end + 1..];
    }
    out
}

fn find_sub(s: &str, pat: &str) -> Option<usize> {
    s.find(pat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_code_blocks() {
        let t = "Here is code:\n```rust\nfn main() {}\n```\nDone.";
        assert_eq!(make_speakable(t), "Here is code: Done.");
    }

    #[test]
    fn code_only_becomes_notice() {
        assert_eq!(make_speakable("```\nfoo bar\n```"), "Code output omitted.");
    }

    #[test]
    fn strips_markdown_and_links() {
        assert_eq!(make_speakable("**Bold** and `x = 1` [docs](https://x.y)"), "Bold and x = 1 docs");
    }

    #[test]
    fn drops_tool_payloads() {
        let t = "I ran the test.\n{\"status\": \"ok\"}\n| a | b |\nAll good.";
        let s = make_speakable(t);
        assert!(s.contains("All good"));
        assert!(!s.contains("status"));
        assert!(!s.contains('|'));
    }
}
