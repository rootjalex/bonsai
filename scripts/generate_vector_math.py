PROLOGUE = "__forceinline__ __host__ __device__"
TYPE_TO_RETURN_TYPE = {"int": "int32_t", "uint": "uint32_t"}
TYPES = ["int", "uint", "float"]
LANES = [2, 3, 4]
FIELDS = ["x", "y", "z", "w"]

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
    input = "v"
    for op, symbol in op_to_symbol.items():
        for type in TYPES:
            for n in LANES:
                return_type = TYPE_TO_RETURN_TYPE[type] if type in TYPE_TO_RETURN_TYPE else type
                function_name = f"{op}({type}{n} {input})"
                expression = symbol.join(
                    [f"{input}.{FIELDS[i]}" for i in range(n)]
                )
                body = f"return {expression};"
                function_definition = f"{PROLOGUE} {return_type} {function_name} {{ {body} }}"
                print(function_definition)
            print()


def generate_vector_idxmax():
    """
    Generates vector reduce (index) functions.
    """
    print("////////////////////////////////////////////////////////////////////////////////")
    print("// idxmax, idxmin")
    print("////////////////////////////////////////////////////////////////////////////////")
    print()
    op_to_symbol = {"idxmax": ">", "idxmin": "<"}
    return_type = INDEX_TYPE
    input = "v"
    for op, symbol in op_to_symbol.items():
        for type in TYPES:
            for n in LANES:
                function_name = f"{op}({type}{n} {input})"
                body = f"""{return_type} idx = 0u;"""
                match n:
                    case 2:
                        body += f"""idx = ({input}.{FIELDS[1]} {symbol} {input}.{FIELDS[0]}) ? 1u : 0u;"""
                    case 3:
                        body += f"""if ({input}.{FIELDS[1]} {symbol} {input}.{FIELDS[0]}) idx = 1u; if ({input}.{FIELDS[2]} {symbol} (idx == 0u ? {input}.{FIELDS[0]} : (idx == 1u ? {input}.{FIELDS[1]} : {input}.{FIELDS[0]}))) idx = 2u;"""
                    case 4:
                        body += f"""if ({input}.{FIELDS[1]} {symbol} {input}.{FIELDS[0]}) idx = 1u; if ({input}.{FIELDS[2]} {symbol} (idx == 0u ? {input}.{FIELDS[0]} : {input}.{FIELDS[1]})) idx = 2u; if ({input}.{FIELDS[3]} {symbol} (idx == 0u ? {input}.{FIELDS[0]} : (idx == 1u ? {input}.{FIELDS[1]} : {input}.{FIELDS[2]}))) idx = 3u;"""
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
            func_name = f" shuffle({vector_type} v, {INDEX_TYPE} indices[{n}])"
            body = f"""{{ {vector_type} r; """
            for i in range(n):
                index_var = f"indices[{i}]"
                body += f"""switch ({index_var}) {{ case 0: r.{FIELDS[i]} = v.x; break; case 1: r.{FIELDS[i]} = v.y; break;"""
                if n > 2:
                    body += f""" case 2: r.{FIELDS[i]} = v.z; break;"""
                if n > 3:
                    body += f""" case 3: r.{FIELDS[i]} = v.w; break;"""
                body += f""" default: abort(); }} """
            body += f"""return r; }}"""
            print(f"{PROLOGUE} {vector_type} {func_name} {body}")
        print()


def generate_vector_less_than_elementwise_operator():
    """Generates elementwise compare vector instructions."""
    print("////////////////////////////////////////////////////////////////////////////////")
    print("// element wise compare")
    print("////////////////////////////////////////////////////////////////////////////////")
    print()

    for type in TYPES:
        for n in LANES:
            vector_type = f"{type}{n}"
            func_name = f"{vector_type} vlt({vector_type} a, {vector_type} b)"
            body = f"""{{{vector_type} r; """
            for i in range(n):
                body += f"""r.{FIELDS[i]} = (a.{FIELDS[i]} < b.{FIELDS[i]}) ? a.{FIELDS[i]} : b.{FIELDS[i]}; """
            body += f"""return r; }}"""
            print(f"{PROLOGUE} {func_name} {body}")
        print()


if __name__ == "__main__":
    generate_vector_less_than_elementwise_operator()
    generate_vector_idxmax()
    generate_vector_reduce()
    generate_vector_shuffle()
