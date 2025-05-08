PROLOGUE = "__forceinline__ __host__ __device__"
TYPE_TO_RETURN_TYPE = {"int": "int32_t", "uint": "uint32_t"}
TYPES = ["int", "uint", "float"]
LANES = [2, 3, 4]
F = ["x", "y", "z", "w"]

INDEX_TYPE = "uint32_t"


def generate_vector_reduce():
    """
    Generates vector reduce (arithmetic) functions.
    """
    print("////////////////////////////////////////////////////////////////////////////////")
    print("// sum")
    print("////////////////////////////////////////////////////////////////////////////////")
    print()
    op_to_symbol = {"sum": "+", "mul": "*"}
    I = "v"
    for op, symbol in op_to_symbol.items():
        for type in TYPES:
            for n in LANES:
                return_type = TYPE_TO_RETURN_TYPE[type] if type in TYPE_TO_RETURN_TYPE else type
                function_name = f"{op}({type}{n} {I})"
                expression = symbol.join(
                    [f"{I}.{F[i]}" for i in range(n)]
                )
                body = f"return {expression};"
                function_definition = f"{PROLOGUE} {return_type} {function_name} {{ {body} }}"
                print(function_definition)
            print()


def generate_vector_idx_minmax():
    """
    Generates vector reduce (index) functions.
    """
    print("////////////////////////////////////////////////////////////////////////////////")
    print("// idxmax, idxmin")
    print("////////////////////////////////////////////////////////////////////////////////")
    print()
    op_to_symbol = {"idxmax": ">", "idxmin": "<"}
    return_type = INDEX_TYPE
    I = "v"
    for op, symbol in op_to_symbol.items():
        for type in TYPES:
            for n in LANES:
                function_name = f"{op}({type}{n} {I})"
                body = f"""{return_type} idx = 0u;"""
                match n:
                    case 2:
                        body += f"""idx = ({I}.{F[1]} {symbol} {I}.{F[0]}) ? 1u : 0u;"""
                    case 3:
                        body += f"""if ({I}.{F[1]} {symbol} {I}.{F[0]}) idx = 1u; if ({I}.{F[2]} {symbol} (idx == 0u ? {I}.{F[0]} : (idx == 1u ? {I}.{F[1]} : {I}.{F[0]}))) idx = 2u;"""
                    case 4:
                        body += f"""if ({I}.{F[1]} {symbol} {I}.{F[0]}) idx = 1u; if ({I}.{F[2]} {symbol} (idx == 0u ? {I}.{F[0]} : {I}.{F[1]})) idx = 2u; if ({I}.{F[3]} {symbol} (idx == 0u ? {I}.{F[0]} : (idx == 1u ? {I}.{F[1]} : {I}.{F[2]}))) idx = 3u;"""
                body += "\n"
                body += "return idx;"
                body += "\n"
                body += "}"
                function_definition = f"{PROLOGUE} {return_type} {function_name} {{\n{body}"
                print(function_definition)
            print()


def generate_vector_shuffle():
    """Generates shuffle vector instructions."""
    print("////////////////////////////////////////////////////////////////////////////////")
    print("// shuffle")
    print("////////////////////////////////////////////////////////////////////////////////")
    print()

    for type in TYPES:
        for n in LANES:
            vector_type = f"{type}{n}"
            func_name = f" shuffle({vector_type} v, std::initializer_list<{INDEX_TYPE}> indices)"
            body = f"""{{ {vector_type} r; auto it = indices.begin();"""
            for i in range(n):
                body += f"""switch (*it++) {{ case 0: r.{F[i]} = v.x; break; case 1: r.{F[i]} = v.y; break;"""
                if n > 2:
                    body += f""" case 2: r.{F[i]} = v.z; break;"""
                if n > 3:
                    body += f""" case 3: r.{F[i]} = v.w; break;"""
                body += f""" default: __builtin_unreachable(); }} """
            body += f"""return r; }}"""
            print(f"{PROLOGUE} {vector_type} {func_name} {body}")
        print()


if __name__ == "__main__":
    generate_vector_idx_minmax()
    generate_vector_reduce()
    generate_vector_shuffle()
