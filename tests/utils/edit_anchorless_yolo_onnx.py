import onnx
import networkx as nx

"""The code below is for editing anchorless YOLO models to fit with the postprocessing function of mblt-model-zoo
Currently, the qubee compiler is under maintenance to deal with the latest YOLO models with transformer architecture.
While doing so, the change of algorithm caused an unexpected change in the supported operation range in anchorless YOLO models.
For this reason, compiling the anchorless YOLO models with the latest qubee may cause conflicts in inference with the mblt-model-zoo.
Thus, it is recommended to edit the pre-trained anchorless YOLO model with this code and compile it with qubee.
We've checked that this code works for yolov3u, v3-sppu, v5(n,s,m,l,x)(6)u, v8(n,s,m,l,x)(-seg, -pose), v9(t,s,m,c,e)(-seg) are supported.
Unfortunately, models with irregular structure, such as yolov3-tinyu, yolov9e, v9e-seg, are not supported.
"""


def build_graph_structure(model):
    """Build a directed graph representation of the ONNX model"""
    G = nx.DiGraph()
    node_map = {}  # Maps output names to nodes

    # Add all nodes and their connections
    for node in model.graph.node:
        node_map[node.name] = node
        for output in node.output:
            G.add_node(output, op_type=node.op_type, node=node)

        # Add edges from inputs to outputs
        for input_name in node.input:
            for output_name in node.output:
                G.add_edge(input_name, output_name)

    return G, node_map


def get_node_by_output_name(model, output_name):
    """Find node that produces the given output"""
    for node in model.graph.node:
        if output_name in node.output:
            return node
    return None


def find_downstream_nodes(G, start_output):
    """Find all nodes downstream from a given output"""
    try:
        descendants = nx.descendants(G, start_output)
        downstream_outputs = list(descendants)
        downstream_outputs.append(start_output)
        return downstream_outputs
    except:
        return [start_output]


def remove_nodes_and_downstream(model, G, start_output):
    """Remove a node and all its downstream nodes"""
    downstream_outputs = find_downstream_nodes(G, start_output)
    outputs_to_remove = set(downstream_outputs)

    # Collect nodes to keep
    nodes_to_keep = [
        node
        for node in model.graph.node
        if not any(output in outputs_to_remove for output in node.output)
    ]

    # Clear the existing nodes using del and add the filtered ones
    del model.graph.node[:]  # Use del instead of clear()
    model.graph.node.extend(nodes_to_keep)

    # Remove from graph
    for output in downstream_outputs:
        if output in G:
            G.remove_node(output)


def process_concat_reshape_shape_removal(model, G):
    """Steps 2 & 3: Remove reshape/shape nodes connected to concat outputs (keep concat)"""
    nodes_to_remove_outputs = []

    # Find reshape and shape nodes that are connected to concat outputs
    for node in model.graph.node:
        if node.op_type in ["Reshape", "Shape"]:
            for input_name in node.input:
                # Check if this input comes from a concat node
                concat_node = get_node_by_output_name(model, input_name)
                if concat_node and concat_node.op_type == "Concat":
                    print(
                        f"Removing {node.op_type} node (connected to concat) and its downstream"
                    )
                    nodes_to_remove_outputs.append(node.output[0])
                    break

    # Remove the reshape/shape nodes and their downstream (but keep concat)
    for output in nodes_to_remove_outputs:
        remove_nodes_and_downstream(model, G, output)


def process_shape_gather_unsqueeze_series(model, G):
    """Step 4: Remove entire shape->gather->unsqueeze series"""
    nodes_to_process = [node for node in model.graph.node if node.op_type == "Shape"]

    for shape_node in nodes_to_process:
        if shape_node not in model.graph.node:  # Skip if already removed
            continue

        shape_output = shape_node.output[0]

        # Find gather node that uses shape output as PRIMARY input
        gather_node = None
        for node in model.graph.node:
            if (
                node.op_type == "Gather"
                and len(node.input) > 0
                and node.input[0] == shape_output
            ):
                gather_node = node
                break

        if gather_node:
            gather_output = gather_node.output[0]

            # Find unsqueeze node that uses gather output as PRIMARY input
            unsqueeze_node = None
            for node in model.graph.node:
                if (
                    node.op_type == "Unsqueeze"
                    and len(node.input) > 0
                    and node.input[0] == gather_output
                ):
                    unsqueeze_node = node
                    break

            if unsqueeze_node:
                print("Removing entire shape->gather->unsqueeze series")
                # Remove starting from the shape node (removes all three and downstream)
                remove_nodes_and_downstream(model, G, shape_output)


def make_end_nodes_outputs(model):
    """Step 5: Make concat/conv nodes at graph end as output nodes while preserving valid original outputs"""
    from onnx import helper, TensorProto

    # Get all available outputs from existing nodes
    available_outputs = set()
    for node in model.graph.node:
        available_outputs.update(node.output)

    # Add graph inputs and initializers as available outputs
    for inp in model.graph.input:
        available_outputs.add(inp.name)
    for init in model.graph.initializer:
        available_outputs.add(init.name)

    # Store only VALID original outputs
    valid_original_outputs = []
    for output in model.graph.output:
        if output.name in available_outputs:
            valid_original_outputs.append(output)
        else:
            print(f"Skipping invalid original output: {output.name}")

    # Find all outputs and inputs in the graph
    all_outputs = set()
    all_inputs = set()

    for node in model.graph.node:
        all_outputs.update(node.output)
        all_inputs.update(node.input)

    # Graph inputs (model inputs and initializers)
    graph_inputs = set(inp.name for inp in model.graph.input)
    graph_inputs.update(init.name for init in model.graph.initializer)

    # End nodes are those whose outputs are not inputs to other nodes
    end_outputs = all_outputs - (all_inputs - graph_inputs)

    # Get names of valid original outputs to avoid duplicates
    original_output_names = set(output.name for output in valid_original_outputs)

    # Clear existing outputs and restore valid originals
    del model.graph.output[:]
    model.graph.output.extend(valid_original_outputs)

    # Add concat/conv end nodes as outputs (only if not already present and they exist)
    for node in model.graph.node:
        if node.op_type in ["Concat", "Conv"]:
            for output in node.output:
                if output in end_outputs and output not in original_output_names:
                    # Verify this output actually exists
                    if output in available_outputs:
                        # Create output value info
                        output_value_info = helper.make_tensor_value_info(
                            output,
                            TensorProto.FLOAT,  # Default to float
                            [],  # Unknown shape
                        )
                        model.graph.output.append(output_value_info)
                        print(
                            f"Added {node.op_type} node output '{output}' as graph output"
                        )


def remove_isolated_subgraphs(model):
    """Step 6: Remove isolated subgraphs not connected to input (enhanced version)"""

    # Build connectivity graph using output names as node identifiers
    G = nx.DiGraph()
    output_to_node = {}  # Maps output names to their producing nodes

    # Add all nodes and map outputs to nodes
    for node in model.graph.node:
        for output in node.output:
            output_to_node[output] = node
            G.add_node(output, node=node, op_type=node.op_type)

    # Add edges based on input/output connections
    for node in model.graph.node:
        for input_name in node.input:
            for output_name in node.output:
                if input_name in G:  # Input exists as an output from another node
                    G.add_edge(input_name, output_name)

    # Get all graph inputs (model inputs + initializers)
    graph_inputs = set()
    for inp in model.graph.input:
        graph_inputs.add(inp.name)
    for init in model.graph.initializer:
        graph_inputs.add(init.name)

    # Find all nodes reachable from graph inputs
    reachable_outputs = set()

    # Start BFS/DFS from each graph input
    for input_name in graph_inputs:
        if input_name in G:
            # Add the input itself
            reachable_outputs.add(input_name)
            # Add all descendants (nodes reachable from this input)
            reachable_outputs.update(nx.descendants(G, input_name))

    # Also include nodes that are directly connected to graph inputs
    for node in model.graph.node:
        if any(inp in graph_inputs for inp in node.input):
            reachable_outputs.update(node.output)
            # Add descendants of these outputs
            for output in node.output:
                if output in G:
                    reachable_outputs.update(nx.descendants(G, output))

    # Preserve nodes that produce original model outputs
    original_output_names = set(output.name for output in model.graph.output)
    for output_name in original_output_names:
        if output_name in G:
            reachable_outputs.add(output_name)
            # Add all ancestors that lead to this output
            reachable_outputs.update(nx.ancestors(G, output_name))

    # Filter nodes: keep only those that produce reachable outputs
    nodes_to_keep = []
    removed_count = 0

    for node in model.graph.node:
        # Check if any of this node's outputs are reachable
        if any(output in reachable_outputs for output in node.output):
            nodes_to_keep.append(node)
        else:
            print(
                f"Removing isolated node: {node.op_type} (outputs: {list(node.output)})"
            )
            removed_count += 1

    print(f"Removed {removed_count} isolated nodes")

    # Update the model
    del model.graph.node[:]
    model.graph.node.extend(nodes_to_keep)


def validate_and_repair_model(model):
    """Validate and repair common issues in the modified model"""

    print("Validating and repairing model...")

    # 1. Clean up invalid output references FIRST
    model = clean_invalid_outputs(model)

    # 2. Remove dangling inputs
    all_available_outputs = set()
    for node in model.graph.node:
        all_available_outputs.update(node.output)

    # Add graph inputs and initializers to available outputs
    for inp in model.graph.input:
        all_available_outputs.add(inp.name)
    for init in model.graph.initializer:
        all_available_outputs.add(init.name)

    # Check for nodes with missing inputs
    nodes_to_remove = []
    for node in model.graph.node:
        for input_name in node.input:
            if input_name and input_name not in all_available_outputs:
                print(
                    f"Node {node.op_type} has missing input '{input_name}', removing node"
                )
                nodes_to_remove.append(node)
                break

    # Remove nodes with missing inputs
    for node in nodes_to_remove:
        model.graph.node.remove(node)

    # 3. Final output validation
    final_available_outputs = set()
    for node in model.graph.node:
        final_available_outputs.update(node.output)
    for inp in model.graph.input:
        final_available_outputs.add(inp.name)
    for init in model.graph.initializer:
        final_available_outputs.add(init.name)

    # Ensure all outputs exist
    for output in model.graph.output:
        if output.name not in final_available_outputs:
            print(f"ERROR: Output '{output.name}' does not exist in graph!")
            return None

    # 4. Clean up unused initializers
    used_initializers = set()
    for node in model.graph.node:
        used_initializers.update(node.input)
    for output in model.graph.output:
        used_initializers.add(output.name)

    initializers_to_keep = []
    for init in model.graph.initializer:
        if init.name in used_initializers:
            initializers_to_keep.append(init)
        else:
            print(f"Removing unused initializer: {init.name}")

    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers_to_keep)

    # 5. Try to infer shapes
    try:
        model = onnx.shape_inference.infer_shapes(model)
        print("Shape inference successful")
    except Exception as e:
        print(f"Shape inference failed: {e}")

    return model


def clean_invalid_outputs(model):
    """Remove invalid output references that don't exist in the graph"""

    # Get all available outputs from existing nodes
    available_outputs = set()
    for node in model.graph.node:
        available_outputs.update(node.output)

    # Add graph inputs (they can also be outputs)
    for inp in model.graph.input:
        available_outputs.add(inp.name)

    # Add initializers (they can also be outputs)
    for init in model.graph.initializer:
        available_outputs.add(init.name)

    # Filter out invalid outputs
    valid_outputs = []
    for output in model.graph.output:
        if output.name in available_outputs:
            valid_outputs.append(output)
        else:
            print(f"Removing invalid output reference: {output.name}")

    # Update the model outputs
    del model.graph.output[:]
    model.graph.output.extend(valid_outputs)

    return model


def process_onnx_model(input_path, output_path):
    """Main function to process ONNX model with all transformations"""
    print(f"Loading ONNX model from: {input_path}")

    # Step 1: Load ONNX model & make it static
    model = onnx.load(input_path)

    # Build graph structure
    G, node_map = build_graph_structure(model)

    print(f"Original model has {len(model.graph.node)} nodes")

    # Step 2 & 3: Remove concat nodes with reshape/shape outputs
    process_concat_reshape_shape_removal(model, G)

    # Rebuild graph after modifications
    G, node_map = build_graph_structure(model)

    # Step 4: Remove shape->gather->unsqueeze series
    process_shape_gather_unsqueeze_series(model, G)

    # Step 5: Make concat/conv end nodes as outputs
    make_end_nodes_outputs(model)

    # Step 6: Remove isolated subgraphs
    remove_isolated_subgraphs(model)

    # Step 7: Clean up invalid outputs BEFORE final validation
    model = clean_invalid_outputs(model)

    # Step 8: Final validation and repair
    model = validate_and_repair_model(model)

    print(f"Processed model has {len(model.graph.node)} nodes")

    # Step 7: Save the modified model
    print(f"Saving processed model to: {output_path}")
    onnx.save(model, output_path)

    # Verify the saved model
    try:
        onnx.checker.check_model(model)
        print("Model validation passed!")
    except Exception as e:
        print(f"Model validation warning: {e}")

    return model


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
    )
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    input_model_path = args.input_path
    assert input_model_path.endswith(".onnx"), "Model should be ONNX"
    if args.output_path is None:
        output_model_path = input_model_path.replace(".onnx", "_modified.onnx")
    else:
        output_model_path = args.output_path
    assert output_model_path.endswith(".onnx"), "Model should be ONNX"

    process_onnx_model(input_model_path, output_model_path)
