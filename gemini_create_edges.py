class KGEdge(BaseModel):
    node1: str
    node2: str
    weight: float
    relation: str

def create_edges_by_gemini(nodes_with_source, papers_dict, model_name="models/gemini-2.5-flash"):
    """
    Use Gemini to suggest edges (relationships) between nodes based on their meaning and context.
    Each edge will be a tuple: (node1, node2, {"weight": 1.0, "relation": relation_string})
    """
    try:
        # Extract just the node names for the prompt
        node_texts = [node[0] for node in nodes_with_source]
        node_list_str = "\n".join(f"- {n}" for n in node_texts)

        paper_texts = "\n\n".join(
            # f"--- Paper: {path} ---\n{text[:4000]}"  # Limit each paper to 4000 chars for prompt size
            f"--- Paper: {path} ---\n{text}"  # include full paper text
            for path, text in (papers_dict or {}).items()
        )
        context_str = f"\n\nContext (paper text excerpts):\n{paper_texts}" if paper_texts else ""

        prompt = f"""
        Given the following list of scientific concepts/nodes from a research paper knowledge graph, provide the most meaningful edges between them based on their relationships.

        For each edge, return an object with:
        - node1: the first node
        - node2: the second node
        - weight: a float indicating the strength of the relationship (e.g. 1.0 for strong, 0.5 for medium, 0.15 for weak, etc.)
        - relation: a short label like 'related_to', 'enables', 'depends_on', etc.

        Only return meaningful and non-trivial relationships. Format as a JSON list.

        Nodes:
        {node_list_str}
        {context_str}        
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[KGEdge],
            },
        )

        print(response.usage_metadata)

        edges = []
        try:
            # Gemini's response.text should be a JSON list of dicts
            edge_objs = json.loads(response.text)
            # Map node text back to (node_text, source) tuples
            node_lookup = {n[0]: n for n in nodes_with_source}
            for edge in edge_objs:
                node1 = node_lookup.get(edge["node1"], (edge["node1"], "unknown"))
                node2 = node_lookup.get(edge["node2"], (edge["node2"], "unknown"))
                attrs = {
                    "weight": edge.get("weight", 1.0),
                    "relation": edge.get("relation", "related_to")
                }
                edges.append((node1, node2, attrs))
        except Exception as e:
            print(f"Error parsing Gemini JSON response: {e}")
            print("Raw response:", response.text)
            return []

        return edges

    except Exception as e:
        print(f"An error occurred with Gemini edge creation (structured): {e}")
        return []