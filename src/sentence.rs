//! Sentence chunker for streaming TTS: emits complete sentences as soon as
//! they are seen, buffers the trailing partial sentence until more text or
//! an explicit flush.

pub struct SentenceSplitter {
    buf: String,
}

impl SentenceSplitter {
    pub fn new() -> Self {
        Self { buf: String::new() }
    }

    /// Feed more text; returns every complete sentence now available.
    pub fn push(&mut self, text: &str) -> Vec<String> {
        self.buf.push_str(text);
        let mut sentences = Vec::new();
        loop {
            match find_sentence_end(&self.buf) {
                Some(end) => {
                    let sentence: String = self.buf.drain(..=end).collect();
                    let sentence = sentence.trim().to_string();
                    if !sentence.is_empty() {
                        sentences.push(sentence);
                    }
                }
                None => break,
            }
        }
        sentences
    }

    /// Emit whatever is left in the buffer (call at end of stream).
    pub fn flush(&mut self) -> Option<String> {
        let rest = self.buf.trim().to_string();
        self.buf.clear();
        if rest.is_empty() {
            None
        } else {
            Some(rest)
        }
    }
}

/// Words after which a dot does not end a sentence.
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "eg", "ie", "am", "pm", "fig", "no", "approx", "inc", "ltd", "p", "m", "a", "u", "s",
];

/// Find the index of the first sentence-ending punctuation that is followed
/// by whitespace or end-of-buffer, skipping common abbreviations and numbers.
fn find_sentence_end(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b != b'.' && b != b'!' && b != b'?' {
            continue;
        }
        // require terminator at end or followed by whitespace
        let followed_by_space = i + 1 == bytes.len() || bytes[i + 1] == b' ' || bytes[i + 1] == b'\n';
        if !followed_by_space {
            continue;
        }
        if b == b'.' && is_abbreviation_or_number(s, i) {
            continue;
        }
        return Some(i);
    }
    None
}

/// Is the dot at byte `i` part of an abbreviation ("Dr.", "p.m.") or a
/// number ("3.14", list marker "1.")?
fn is_abbreviation_or_number(s: &str, i: usize) -> bool {
    let bytes = s.as_bytes();
    // decimal: digit before and after
    if i > 0 && bytes[i - 1].is_ascii_digit() && bytes.get(i + 1).is_some_and(|b| b.is_ascii_digit()) {
        return true;
    }
    // list marker: "1. text" or "1.5" style enumeration
    if i > 0
        && bytes[i - 1].is_ascii_digit()
        && bytes.get(i + 2).is_some_and(|b| b.is_ascii_digit())
    {
        return true;
    }
    // walk back over letters and dots to find the word before the dot
    let mut start = i;
    while start > 0 {
        let c = bytes[start - 1];
        if c.is_ascii_alphabetic() || c == b'.' {
            start -= 1;
        } else {
            break;
        }
    }
    let word = s[start..i].replace('.', "").to_ascii_lowercase();
    word.len() == 1 || ABBREVIATIONS.contains(&word.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_sentences() {
        let mut sp = SentenceSplitter::new();
        let got = sp.push("Hello there. How are");
        assert_eq!(got, vec!["Hello there.".to_string()]);
        assert_eq!(sp.push(" you? Fine!"), vec!["How are you?".to_string(), "Fine!".to_string()]);
        assert_eq!(sp.flush(), None);
    }

    #[test]
    fn flushes_partial() {
        let mut sp = SentenceSplitter::new();
        assert!(sp.push("partial sentence").is_empty());
        assert_eq!(sp.flush().as_deref(), Some("partial sentence"));
    }

    #[test]
    fn keeps_abbreviations() {
        let mut sp = SentenceSplitter::new();
        let got = sp.push("Meet Dr. Smith at 5 p.m. today");
        assert!(got.is_empty(), "should not split on abbreviations, got {:?}", got);
    }
}
