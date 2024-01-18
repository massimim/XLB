import warp as wp


class Container(object):
    def __init__(self, name, grid, function, parameter_list):
        self.id = id
        self.grid = grid
        self.function = function
        self.parameter_list = parameter_list
        self.name = name

        self.generate_warp_kernel()


    def generate_warp_kernel(self):
        proccessed_parameter_list = []
        for parameter in self.parameter_list:
            proccessed_parameter_list.append(parameter.get_partition(0))

        parameter_types = [type(param) for param in proccessed_parameter_list]

        # Dynamically create the function signature with full type names (including namespaces)
        func_args = ', '.join([f'arg{i}: {parameter_types[i].__module__}.{parameter_types[i].__qualname__}' for i in
                               range(len(parameter_types))])
        func_body = f"@wp.kernel\n"
        func_body += f"def myfoo({func_args}):\n"
        func_body += "    # Perform additional operations here\n"
        func_body += f"    return func({', '.join(['arg' + str(i) for i in range(len(parameter_types))])})\n"

        print(func_body)
        pass

    def run(self):
        return self.id

    @staticmethod
    def kernel(self, function, span, parameter_list):
        proccessed_parameter_list = []
        for parameter in parameter_list:
            proccessed_parameter_list.append(parameter.get_partition())

        @wp.kernel
        def kernel(span, *proccessed_parameter_list):
            x, y, z = wp.tid()
            idx = span.get_idx()
            function(idx, *proccessed_parameter_list)
            pass

# Compare this snippet from experimental/neon/parameter.py:
