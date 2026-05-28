use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::inference::runner::LlamaRunner;

#[pyclass(name = "AetherModel")]
pub struct PyAetherModel {
    runner: Arc<Mutex<LlamaRunner>>,
}

#[pymethods]
impl PyAetherModel {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let runner = LlamaRunner::from_gguf(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;
        Ok(PyAetherModel {
            runner: Arc::new(Mutex::new(runner)),
        })
    }

    fn generate(
        &self,
        prompt: &str,
        max_tokens: i32,
        temperature: f32,
        top_p: f32,
    ) -> PyResult<String> {
        let mut runner = self
            .runner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Internal lock error: {}", e)))?;
        runner
            .generate(prompt, max_tokens as usize, temperature, top_p, 1.0)
            .map_err(|e| PyRuntimeError::new_err(format!("Generation failed: {}", e)))
    }

    fn stream(
        &self,
        prompt: &str,
        max_tokens: i32,
        temperature: f32,
        top_p: f32,
    ) -> PyResult<TokenStream> {
        let (tx, rx) = mpsc::channel::<String>();

        let runner = self.runner.clone();
        let prompt = prompt.to_string();

        thread::spawn(move || {
            let mut r = match runner.lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            let _ = r.generate_callback(
                &prompt,
                max_tokens as usize,
                temperature,
                top_p,
                1.0,
                |_| {},
                |t| {
                    let _ = tx.send(t.to_string());
                },
            );
        });

        Ok(TokenStream { rx: Mutex::new(rx) })
    }

    fn __repr__(&self) -> String {
        "AetherModel".to_string()
    }
}

#[pyclass(name = "TokenStream")]
pub struct TokenStream {
    rx: Mutex<mpsc::Receiver<String>>,
}

#[pymethods]
impl TokenStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRefMut<'_, Self>) -> PyResult<Option<String>> {
        let rx = slf
            .rx
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Internal lock error: {}", e)))?;
        match rx.recv_timeout(std::time::Duration::from_secs(300)) {
            Ok(token) => Ok(Some(token)),
            Err(mpsc::RecvTimeoutError::Timeout) => Err(PyRuntimeError::new_err(
                "No token received within 300 seconds",
            )),
            Err(mpsc::RecvTimeoutError::Disconnected) => Ok(None),
        }
    }
}

pub fn aether_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAetherModel>()?;
    m.add_class::<TokenStream>()?;
    Ok(())
}
