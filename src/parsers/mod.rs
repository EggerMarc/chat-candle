pub mod json;
pub mod reasoning;
pub mod structured;
pub mod tool;

/// Length of the longest suffix of `s` that is a proper prefix of `marker`.
/// Lets incremental scanners hold back a partial marker split across chunk
/// boundaries (e.g. `<tool_ca` then `ll>`).
pub(crate) fn partial_suffix_len(s: &str, marker: &str) -> usize {
    let max = marker.len().min(s.len());
    for k in (1..=max).rev() {
        let start = s.len() - k;
        if s.is_char_boundary(start) && marker.as_bytes().starts_with(&s.as_bytes()[start..]) {
            return k;
        }
    }
    0
}
