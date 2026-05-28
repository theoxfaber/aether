pub mod tensor_types {
    use std::fmt;

    /// Globally unique identifier for a tensor node in the computation graph.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TensorId(pub u64);

    impl TensorId {
        /// Allocate a new globally unique ID.
        pub fn next() -> Self {
            static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            TensorId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
        }
    }

    /// Element data type for tensor values.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    pub enum Dtype {
        F32,
        F16,
        BF16,
    }

    impl fmt::Display for Dtype {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Dtype::F32 => write!(f, "f32"),
                Dtype::F16 => write!(f, "f16"),
                Dtype::BF16 => write!(f, "bf16"),
            }
        }
    }

    /// Storage for tensor data in either F32 or F16 format.
    #[derive(Debug, Clone)]
    pub enum AnyData {
        F32(Vec<f32>),
        F16(Vec<half::f16>),
    }

    impl AnyData {
        pub fn as_f32_slice(&self) -> &[f32] {
            match self {
                AnyData::F32(v) => v,
                AnyData::F16(_v) => {
                    panic!("Tensor data is in F16 format; call to_f32() to convert");
                }
            }
        }

        pub fn to_f32(&self) -> Vec<f32> {
            match self {
                AnyData::F32(v) => v.clone(),
                AnyData::F16(v) => v.iter().map(|&x| f32::from(x)).collect(),
            }
        }

        pub fn len(&self) -> usize {
            match self {
                AnyData::F32(v) => v.len(),
                AnyData::F16(v) => v.len(),
            }
        }

        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }
    }

    /// Multi-dimensional shape describing a tensor's layout.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Shape {
        pub(crate) dims: Vec<usize>,
    }

    impl Shape {
        /// Create a shape from dimension sizes.
        pub fn new(dims: Vec<usize>) -> Self {
            Self { dims }
        }

        pub fn dims(&self) -> &[usize] {
            &self.dims
        }

        pub fn ndim(&self) -> usize {
            self.dims.len()
        }

        pub fn num_elements(&self) -> usize {
            if self.dims.is_empty() {
                0
            } else {
                self.dims.iter().product()
            }
        }
    }

    /// A node in the computation graph holding data, shape, and dtype.
    #[derive(Debug, Clone)]
    pub struct Tensor {
        pub(crate) id: TensorId,
        pub(crate) data: AnyData,
        pub(crate) shape: Shape,
        pub(crate) dtype: Dtype,
    }

    impl Tensor {
        /// Create a new F32 tensor from flat data and shape.
        /// Panics if `data.len() != shape.num_elements()`.
        pub fn new(data: Vec<f32>, shape: Shape) -> Self {
            assert_eq!(
                data.len(),
                shape.num_elements(),
                "Data length does not match shape size"
            );
            Self {
                id: TensorId::next(),
                data: AnyData::F32(data),
                shape,
                dtype: Dtype::F32,
            }
        }

        pub fn new_with_data(data: AnyData, shape: Shape, dtype: Dtype) -> Self {
            let len = data.len();
            assert_eq!(
                len,
                shape.num_elements(),
                "Data length does not match shape size"
            );
            Self {
                id: TensorId::next(),
                data,
                shape,
                dtype,
            }
        }

        pub fn with_dtype(mut self, dtype: Dtype) -> Self {
            self.dtype = dtype;
            self
        }

        pub fn id(&self) -> TensorId {
            self.id
        }

        pub fn data(&self) -> &[f32] {
            self.data.as_f32_slice()
        }

        pub fn data_raw(&self) -> &AnyData {
            &self.data
        }

        pub fn into_data_raw(self) -> AnyData {
            self.data
        }

        pub fn shape(&self) -> &Shape {
            &self.shape
        }

        pub fn dtype(&self) -> Dtype {
            self.dtype
        }
    }

    /// A tensor that may live on GPU without a CPU copy
    pub struct GpuTensor {
        pub id: TensorId,
        pub shape: Shape,
        pub dtype: Dtype,
        /// Only Some if explicitly read back
        cpu_cache: Option<Vec<f32>>,
    }

    impl GpuTensor {
        pub fn new(id: TensorId, shape: Shape) -> Self {
            Self {
                id,
                shape,
                dtype: Dtype::F32,
                cpu_cache: None,
            }
        }

        /// Force readback to CPU. Expensive — only call when user needs Vec<f32>.
        pub fn to_cpu(
            &mut self,
            registry: &crate::memory::registry::BufferRegistry,
            device: &wgpu::Device,
        ) -> Result<&[f32], crate::Error> {
            if self.cpu_cache.is_none() {
                self.cpu_cache = Some(registry.ensure_cpu(self.id, Some(device))?);
            }
            Ok(self.cpu_cache.as_ref().unwrap())
        }
    }
}
pub use tensor_types::{AnyData, Dtype, GpuTensor, Shape, Tensor, TensorId};
