use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

/// A single trace event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TraceEvent {
    pub name: String,
    pub category: String,
    pub start_us: u64,
    pub dur_us: u64,
    pub thread_id: u64,
    pub args: HashMap<String, serde_json::Value>,
}

/// Execution trace recorder. Collects span-based timing events
/// during graph execution. Thread-safe, append-only.
pub struct TraceRecorder {
    events: Mutex<Vec<TraceEvent>>,
    start: Instant,
}

impl Default for TraceRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceRecorder {
    pub fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            start: Instant::now(),
        }
    }

    fn now_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }

    /// Record a named span. Returns a guard that records the end on drop.
    pub fn span(&self, name: &str, category: &str) -> SpanGuard<'_> {
        let ts = self.now_us();
        SpanGuard {
            recorder: self,
            name: name.to_string(),
            category: category.to_string(),
            start_us: ts,
        }
    }

    /// Record an instant event (zero-duration).
    pub fn event(&self, name: &str, category: &str) {
        let ts = self.now_us();
        self.record(name, category, ts, 0, HashMap::new());
    }

    /// Record a complete event.
    pub fn record(
        &self,
        name: &str,
        category: &str,
        start_us: u64,
        dur_us: u64,
        args: HashMap<String, serde_json::Value>,
    ) {
        let mut events = self.events.lock().unwrap();
        events.push(TraceEvent {
            name: name.to_string(),
            category: category.to_string(),
            start_us,
            dur_us,
            thread_id: 0,
            args,
        });
    }

    /// Export as Chrome Trace JSON (loadable in chrome://tracing).
    pub fn export_chrome_trace(&self) -> String {
        let events = self.events.lock().unwrap();
        let mut out = String::from("[");
        for (i, e) in events.iter().enumerate() {
            if i > 0 {
                out.push_str(",\n");
            }
            out.push_str(&format!(
                r#"{{"name":"{}","cat":"{}","ph":"X","ts":{},"dur":{},"pid":1,"tid":{},"args":{}}}"#,
                e.name, e.category, e.start_us, e.dur_us, e.thread_id,
                serde_json::to_string(&e.args).unwrap_or_default()
            ));
        }
        out.push(']');
        out
    }

    pub fn num_events(&self) -> usize {
        self.events.lock().unwrap().len()
    }

    /// Reset all events.
    pub fn reset(&self) {
        let mut events = self.events.lock().unwrap();
        events.clear();
    }
}

/// Drop guard that records end timestamp.
pub struct SpanGuard<'a> {
    recorder: &'a TraceRecorder,
    name: String,
    category: String,
    start_us: u64,
}

impl Drop for SpanGuard<'_> {
    fn drop(&mut self) {
        let dur = self.recorder.start.elapsed().as_micros() as u64 - self.start_us;
        self.recorder.record(
            &self.name,
            &self.category,
            self.start_us,
            dur,
            HashMap::new(),
        );
    }
}
