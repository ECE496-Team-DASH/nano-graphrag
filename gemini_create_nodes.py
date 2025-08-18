

class KGNode(BaseModel):
    node: str

def gemini_create_nodes(paper_text: str, node_limit: int, paper_source: str) -> list:
    """
    Extract high-level scientific concepts using the Gemini 2.5 Flash model (google-genai library, correct API).

    Parameters:
    -----------
    paper_text : str
        The text content of the paper.
    node_limit : int
        The maximum number of concepts to extract.
    paper_source : str
        The source identifier for the paper.

    Returns:
    --------
    list
        A list of (concept, paper_source) tuples.
    """
    if not gemini_api_key:
        print("Gemini API key is not configured. Skipping node creation.")
        return []

    try:
        model_name = "gemini-2.5-flash"
        prompt = f"""
        From the following research paper text, extract the top {node_limit} most important high-level scientific concepts, methods, and results.\nFocus on concepts that are central to the paper's contribution. \n\nPaper Text:\n\"\"\"\n{paper_text}\n\"\"\"
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[KGNode],
            },
        )

        print(response.usage_metadata)

        # Parse response.text as JSON and extract nodes
        try:
            node_objs = json.loads(response.text)
            # node_objs should be a list of dicts with a 'node' key
            concepts = [obj["node"].strip() for obj in node_objs if "node" in obj and obj["node"].strip()]
        except Exception as e:
            print(f"Error parsing Gemini JSON response: {e}")
            print("Raw response:", response.text)
            return []

        nodes_with_source = [(concept, paper_source) for concept in concepts]
        return nodes_with_source
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return []
