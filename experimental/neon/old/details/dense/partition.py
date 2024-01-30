import neon as ne
import warp as wp
from typing import TypeVar, Generic
import dense as neon_dense


def make_partition(dtype,
                   layout_dim: ne.Index,
                   memory):
    dtype_str = f"{dtype.__qualname__}"
    # if f"{dtype.__module__}" == "builtins":
    #     dtype_str = f"{dtype.__qualname__}"

    class_name = f"Partition_{dtype_str}".replace(".", "_")

    template_generation = f"""  
@wp.struct
class {class_name}:
    layout_dim: ne.Index
    memory: wp.array(dtype={dtype_str})
    default_value: {dtype_str}

    def set(self,
            layout_dim: ne.Index,
            memory: wp.array(dtype={dtype_str})):
        self.layout_dim = layout_dim
        self.memory = memory
    
    def gerTypeName(self):
        return "{dtype_str}"
        
    def message(self):
        return "Partition_{dtype_str}"
    
    def operator_brackets(self, idx: neon_dense.Index) :
        pitch_z: int = self.layout_dim.x * self.layout_dim.y
        pitch_y: int = self.layout_dim.x
        return self.memory[idx.x + idx.y * pitch_y + idx.z * pitch_z]
        
    #
    # def setDefautlValue(self, value: dtype):
    #     self.default_value = value
    #

    """
    exec(template_generation, globals(), locals())
    PartitionClass = locals()[class_name]
    p = PartitionClass()
    p.set(layout_dim, memory)
    # Create and return an instance of the dynamically created class
    return p