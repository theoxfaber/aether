use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::loader::gguf::GGUFModel;
use crate::Error;

use super::model_loader::LlamaModel;

// ── Architecture trait ────────────────────────────────────────────────────

/// A loader that knows how to convert raw GGUF tensors into model weights
/// for a specific architecture family (e.g. Llama, Mistral, Qwen2, etc.).
pub trait ArchitectureLoader: Send + Sync {
    /// The `general.architecture` value this loader handles (e.g. `"llama"`).
    fn name(&self) -> &'static str;

    /// Load a [`LlamaModel`] from parsed GGUF data.
    fn load(
        &self,
        model: &GGUFModel,
        lazy_layers: bool,
    ) -> Result<LlamaModel, Error>;
}

// ── Registry ──────────────────────────────────────────────────────────────

type LoaderMap = HashMap<&'static str, Box<dyn ArchitectureLoader>>;

static REGISTRY: OnceLock<Mutex<LoaderMap>> = OnceLock::new();

fn registry() -> &'static Mutex<LoaderMap> {
    REGISTRY.get_or_init(|| {
        let mut m: LoaderMap = HashMap::new();
        for name in &["llama", "mistral", "phi3", "qwen2", "gemma2", "deepseek2"] {
            m.insert(*name, Box::new(LlamaLoader));
        }
        Mutex::new(m)
    })
}

/// Register an architecture loader for a single name.
/// Panics if a loader for that name is already registered.
pub fn register_loader(loader: Box<dyn ArchitectureLoader>) {
    let name = loader.name();
    let mut reg = registry().lock().expect("ArchitectureLoader registry lock poisoned");
    if reg.contains_key(name) {
        panic!("ArchitectureLoader for '{}' already registered", name);
    }
    reg.insert(name, loader);
}

/// Load a model using the registered loader for `architecture_name`.
///
/// Returns `None` when no loader is registered for that architecture
/// (callers should fall back to the default Llama loader).
pub fn load_model(
    architecture_name: &str,
    model: &GGUFModel,
    lazy_layers: bool,
) -> Option<Result<LlamaModel, Error>> {
    let reg = registry().lock().expect("ArchitectureLoader registry lock poisoned");
    reg.get(architecture_name).map(|loader| loader.load(model, lazy_layers))
}

// ── Default Llama-family loader ───────────────────────────────────────────

struct LlamaLoader;

impl ArchitectureLoader for LlamaLoader {
    fn name(&self) -> &'static str {
        "llama"
    }

    fn load(
        &self,
        model: &GGUFModel,
        lazy_layers: bool,
    ) -> Result<LlamaModel, Error> {
        LlamaModel::from_gguf(model, lazy_layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_architectures_registered() {
        for name in &["llama", "mistral", "phi3", "qwen2", "gemma2", "deepseek2"] {
            assert!(
                registry().lock().unwrap().contains_key(name),
                "loader for '{}' should be registered",
                name
            );
        }
    }

    #[test]
    fn test_unknown_architecture_returns_none() {
        let model = GGUFModel {
            metadata: std::collections::HashMap::new(),
            tensors: std::collections::HashMap::new(),
        };
        assert!(load_model("nonexistent_arch_v42", &model, false).is_none());
    }

    struct TestLoader;
    impl ArchitectureLoader for TestLoader {
        fn name(&self) -> &'static str {
            "test_arch"
        }
        fn load(
            &self,
            _model: &GGUFModel,
            _lazy_layers: bool,
        ) -> Result<LlamaModel, Error> {
            Err(Error::ExecutionError("test only".into()))
        }
    }

    #[test]
    fn test_register_new_loader() {
        register_loader(Box::new(TestLoader));
        assert!(registry().lock().unwrap().contains_key("test_arch"));
    }
}
