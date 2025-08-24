"""Enhanced Knowledge Graph Generation with Comprehensive LLM Support"""

import networkx as nx
from pyvis.network import Network
import json
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from sentence_transformers import SentenceTransformer, util
import numpy as np

from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMProvider:
    """
    Simple abstraction for LLM providers to allow easy model switching
    """
    def __init__(self, provider_type="gemini", api_key=None, model_name=None):
        self.provider_type = provider_type
        self.api_key = api_key
        self.model_name = model_name or self._get_default_model()
        self.client = None
        self._initialize_client()
    
    def _get_default_model(self):
        if self.provider_type == "gemini":
            return "gemini-2.5-flash"
        return "default-model"
    
    def _initialize_client(self):
        if self.provider_type == "gemini" and self.api_key:
            try:
                import google.genai as genai
                self.client = genai.Client(api_key=self.api_key)
            except ImportError:
                print("google-genai library not available. Install with: pip install google-genai")
                self.client = None
    
    def generate_text(self, prompt: str) -> str:
        """Generate text response from the LLM"""
        if self.provider_type == "gemini":
            return self._generate_gemini(prompt)
        else:
            raise NotImplementedError(f"Provider {self.provider_type} not implemented")
    
    def generate_structured(self, prompt: str, schema=None) -> str:
        """Generate structured response (JSON) from the LLM"""
        if self.provider_type == "gemini":
            return self._generate_gemini_structured(prompt, schema)
        else:
            raise NotImplementedError(f"Provider {self.provider_type} not implemented")
    
    def _generate_gemini(self, prompt: str) -> str:
        if not self.client:
            raise ValueError("Gemini client not initialized")
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text.strip()
    
    def _generate_gemini_structured(self, prompt: str, schema=None) -> str:
        if not self.client:
            raise ValueError("Gemini client not initialized")
        
        config = {}
        if schema:
            config = {
                "response_mime_type": "application/json",
                "response_schema": schema,
            }
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        return response.text.strip()


class KGNode(BaseModel):
    node: str


class KGEdge(BaseModel):
    node1: str
    node2: str
    weight: float
    relation: str


@dataclass
class ArxivPaper:
    """Enhanced ArXiv paper representation"""
    id: str
    title: str
    authors: List[str]
    summary: str
    published: datetime
    updated: datetime
    categories: List[str]
    pdf_url: str
    relevance_score: float = 0.0
    is_recent: bool = False
    download_size: Optional[str] = None


class EnhancedKG:
    """Enhanced Knowledge Graph Generation replacing old Gemini methods"""
    
    def __init__(self, llm_provider="gemini", model_name=None):
        # Load environment variables from .env file
        load_dotenv()

        # Initialize LLM provider
        api_key = os.getenv("GEMINI_API_KEY") if llm_provider == "gemini" else None
        self.llm = LLMProvider(provider_type=llm_provider, api_key=api_key, model_name=model_name)
        
        if api_key:
            print(f"{llm_provider.capitalize()} API key configured.")
        else:
            print(f"{llm_provider.capitalize()} API key not found. Please set it in the .env file.")

    def summarize_paper(self, paper_text: str, paper_source: str, max_chars: int = 4000) -> str:
        """
        Summarize a research paper focusing on key scientific concepts and contributions.
        """
        if not self.llm.client:
            print("LLM client is not configured. Returning truncated text.")
            return paper_text[:max_chars]
            
        try:
            prompt = f"""
            Summarize the following research paper with a focus on:
            
            1. Main research objectives and questions
            2. Key methodologies and approaches used
            3. Important findings and results
            4. Core scientific concepts and terminology
            5. Novel contributions to the field
            
            Keep the summary under {max_chars} characters while preserving all important scientific concepts and technical terms.
            
            Paper Text:
            \"\"\"{paper_text}\"\"\"
            """
            
            summary = self.llm.generate_text(prompt)
            
            # Enforce character limit
            if len(summary) > max_chars:
                summary = summary[:max_chars]
                
            print(f"Summarized {paper_source} to {len(summary)} chars.")
            return summary
            
        except Exception as e:
            print(f"Error summarizing {paper_source}: {e}")
            return paper_text[:max_chars]

    def create_nodes(self, paper_summary: str, node_limit: int, paper_source: str) -> list:
        """
        Extract high-level scientific concepts using the configured LLM.
        Compatible with existing nano-graphrag interface.
        """
        if not self.llm.client:
            print("LLM client is not configured. Skipping node creation.")
            return []

        try:
            prompt = f"""
            From the following research paper summary, extract the top {node_limit} most important high-level scientific concepts, methods, and results.
            Focus on concepts that are central to the paper's contribution.
            
            Paper Summary:
            \"\"\"{paper_summary}\"\"\"
            """

            response_text = self.llm.generate_structured(prompt, list[KGNode])

            # Parse response as JSON and extract nodes
            try:
                node_objs = json.loads(response_text)
                concepts = [obj["node"].strip() for obj in node_objs if "node" in obj and obj["node"].strip()]
            except Exception as e:
                print(f"Error parsing LLM JSON response: {e}")
                print("Raw response:", response_text)
                return []

            nodes_with_source = [(concept, paper_source) for concept in concepts]
            return nodes_with_source
        except Exception as e:
            print(f"An error occurred with the LLM API: {e}")
            return []

    def create_edges(self, nodes_with_source, summarized_papers):
        """
        Use LLM to suggest edges (relationships) between nodes based on their meaning and context.
        Compatible with existing nano-graphrag interface.
        """
        try:
            # Extract just the node names for the prompt
            node_texts = [node[0] for node in nodes_with_source]
            node_list_str = "\n".join(f"- {n}" for n in node_texts)

            paper_texts = "\n\n".join(
                f"--- Paper: {path} ---\n{summary}"
                for path, summary in summarized_papers.items()
            )
            context_str = f"\n\nContext (summarized papers):\n{paper_texts}" if paper_texts else ""

            prompt = f"""
            Given the following list of scientific concepts/nodes from a research paper knowledge graph, provide the most meaningful edges between them based on their relationships.

            For each edge, return an object with:
            - node1: the first node
            - node2: the second node
            - weight: a float indicating the strength of the relationship (e.g. 1.0 for strong, 0.5 for medium, 0.15 for weak, etc.)
            - relation: a short label like 'related_to', 'enables', 'depends_on', etc.

            Only return meaningful and non-trivial relationships. There should be no orphan nodes, it should be a connected graph. Format as a JSON list.

            Nodes:
            {node_list_str}
            {context_str}        
            """

            response_text = self.llm.generate_structured(prompt, list[KGEdge])

            edges = []
            try:
                # LLM's response should be a JSON list of dicts
                edge_objs = json.loads(response_text)
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
                print(f"Error parsing LLM JSON response: {e}")
                print("Raw response:", response_text)
                return []

            return edges

        except Exception as e:
            print(f"An error occurred with LLM edge creation: {e}")
            return []   

    def ensure_graph_connectivity(self, nodes_with_source, edges):
        """
        Ensure the graph is connected by adding edges between disconnected components.
        Uses semantic similarity to find the best connections between components.
        """
        # Create a temporary graph to analyze connectivity
        temp_graph = nx.Graph()
        
        # Add nodes
        for node_text, source in nodes_with_source:
            temp_graph.add_node(node_text)
        
        # Add existing edges
        for (node1_with_source, node2_with_source, attrs) in edges:
            node1_text, _ = node1_with_source
            node2_text, _ = node2_with_source
            temp_graph.add_edge(node1_text, node2_text)
        
        # Find connected components
        components = list(nx.connected_components(temp_graph))
        print(f"Found {len(components)} connected components")
        
        if len(components) <= 1:
            print("Graph is already connected!")
            return edges
        
        # Initialize sentence transformer for semantic similarity
        print("Loading sentence transformer for connectivity analysis...")
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading sentence transformer: {e}")
            return edges
        
        # Connect components by finding best semantic matches
        additional_edges = []
        main_component = max(components, key=len)  # Largest component
        
        for component in components:
            if component == main_component:
                continue
                
            # Find best connection between this component and main component
            best_similarity = -1
            best_edge = None
            
            # Get embeddings for nodes in both components
            component_nodes = list(component)
            main_nodes = list(main_component)
            
            # Compute embeddings
            component_embeddings = model.encode(component_nodes)
            main_embeddings = model.encode(main_nodes)
            
            # Find best similarity match
            similarities = util.pytorch_cos_sim(component_embeddings, main_embeddings)
            
            # Find the highest similarity
            max_sim_idx = np.unravel_index(similarities.argmax(), similarities.shape)
            best_similarity = similarities[max_sim_idx].item()
            
            # Create edge between most similar nodes
            component_node = component_nodes[max_sim_idx[0]]
            main_node = main_nodes[max_sim_idx[1]]
            
            # Find the node_with_source tuples for these nodes
            component_node_with_source = next((n for n in nodes_with_source if n[0] == component_node), None)
            main_node_with_source = next((n for n in nodes_with_source if n[0] == main_node), None)
            
            # Skip if we can't find the nodes in the original list
            if component_node_with_source is None or main_node_with_source is None:
                print(f"  Warning: Could not find nodes '{component_node}' or '{main_node}' in original node list")
                continue
            
            edge_attrs = {
                "weight": max(0.1, best_similarity * 0.5),  # Lower weight for connectivity edges
                "relation": "semantic_similarity"
            }
            
            additional_edges.append((component_node_with_source, main_node_with_source, edge_attrs))
            print(f"  Connecting '{component_node}' to '{main_node}' (similarity: {best_similarity:.3f})")
        
        print(f"Added {len(additional_edges)} connectivity edges")
        return edges + additional_edges

    def create_graph(self, nodes_with_source, edges, paper_colors=None, ensure_connectivity=True):
        """Create NetworkX graph from nodes and edges"""
        graph = nx.Graph()
        
        # If no colors provided, generate random distinct colors
        if paper_colors is None:
            import random
            all_sources = set(source for _, source in nodes_with_source)
            paper_colors = {source: f"#{random.randint(0, 0xFFFFFF):06x}" for source in all_sources}

        # Add nodes with source information as attributes
        for node_text, source in nodes_with_source:
            graph.add_node(node_text, source=source, color=paper_colors.get(source, "#808080"), 
                        title=f"Source: {source}")

        # Add edges (store both weight and relation if present)
        for (node1_with_source, node2_with_source, attrs) in edges:
            node1_text, _ = node1_with_source
            node2_text, _ = node2_with_source
            edge_attrs = {"weight": attrs.get("weight", 1)}
            if "relation" in attrs:
                edge_attrs["relation"] = attrs["relation"]
            graph.add_edge(node1_text, node2_text, **edge_attrs)

        print(f"Number of nodes: {len(graph.nodes)}")
        print(f"Number of edges: {len(graph.edges)}")
        print(f"Sources: {set(nx.get_node_attributes(graph, 'source').values())}")

        return graph

    def export_graph_to_dashkg_json(self, graph, output_path):
        """
        Export knowledge graph data to a .dashkg.json file format
        """
        # Prepare nodes data from graph
        nodes_data = []
        for node_text, data in graph.nodes(data=True):
            node_info = {
                "id": node_text,
                "label": node_text,
                "source": data.get('source', 'Unknown'),
                "color": data.get('color', '#808080'),
                "degree": graph.degree(node_text)
            }
            nodes_data.append(node_info)
        
        # Prepare edges data from graph
        edges_data = []
        for source, target, data in graph.edges(data=True):
            edge_info = {
                "source": source,
                "target": target,
                "weight": data.get('weight', 1.0),
                "relation": data.get('relation', 'related_to')
            }
            edges_data.append(edge_info)
        
        # Prepare graph metadata
        sources = set(nx.get_node_attributes(graph, 'source').values())
        metadata = {
            "total_nodes": len(graph.nodes),
            "total_edges": len(graph.edges),
            "sources": list(sources),
            "connected_components": len(list(nx.connected_components(graph))),
            "graph_density": nx.density(graph),
            "average_degree": sum(dict(graph.degree()).values()) / len(graph.nodes) if graph.nodes else 0
        }
        
        # Combine all data
        dashkg_data = {
            "metadata": metadata,
            "nodes": nodes_data,
            "edges": edges_data,
            "format_version": "1.0"
        }
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dashkg_data, f, indent=2, ensure_ascii=False)
        
        print(f"Knowledge graph data exported to: {output_path}")
        return output_path

    def advanced_graph_to_html(self, graph, path: str, display: bool = False):
        """
        Create an enhanced interactive HTML visualization of the knowledge graph
        """
        net = Network(height="750px", width="100%", notebook=True, cdn_resources="in_line")
        
        # Configure advanced network options for better visualization
        net.set_options("""
        {
        "nodes": {
            "font": {"size": 14, "face": "Tahoma"},
            "scaling": {
            "min": 10,
            "max": 30,
            "label": {
                "enabled": true,
                "min": 14,
                "max": 30
            }
            },
            "shape": "dot"
        },
        "edges": {
            "font": {"size": 12, "align": "middle"},
            "color": {"inherit": "both"},
            "smooth": {"type": "continuous", "roundness": 0.5},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
            "scaling": {"min": 1, "max": 10}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "hideEdgesOnDrag": false,
            "navigationButtons": true
        },
        "physics": {
            "barnesHut": {
            "gravitationalConstant": -80000,
            "springLength": 250,
            "springConstant": 0.001,
            "damping": 0.09
            },
            "minVelocity": 0.75
        }
        }
        """)
        
        # Add nodes with source information and size based on degree
        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
        
        for node, data in graph.nodes(data=True):
            # Scale node size based on its degree centrality
            size = 10 + (degrees.get(node, 1) / max_degree) * 20
            
            # Create informative tooltip
            source = data.get('source', 'Unknown')
            paper_name = source.split('/')[-1]
            title = f"Concept: {node}<br>Source: {paper_name}<br>Connections: {degrees.get(node, 0)}"
            
            net.add_node(
                node, 
                color=data.get('color', '#808080'), 
                title=title, 
                label=node,
                size=size
            )
        
        # Add edges with relationship information when available
        for source, target, data in graph.edges(data=True):
            weight = data.get('weight', 1)
            relation = data.get('relation', None)
            if relation:
                title = f"Relation: {relation}<br>Weight: {weight}"
                net.add_edge(
                    source, target, 
                    value=min(weight, 10),  # Cap weight for visualization
                    title=title,
                    label=relation
                )
            else:
                title = f"Weight: {weight}"
                net.add_edge(source, target, value=min(weight, 10), title=title)
        
        # Add legend for paper sources
        paper_sources = set(nx.get_node_attributes(graph, 'source').values())
        legend_html = "<div>"
        legend_html += "<h3>Paper Sources</h3>"
        
        for source in paper_sources:
            paper_name = source.split('/')[-1]
            color = next((data.get('color') for _, data in graph.nodes(data=True) if data.get('source') == source), '#808080')
            legend_html += f"<div><span style='background-color:{color}; width:15px; height:15px; display:inline-block; margin-right:5px;'></span>{paper_name}</div>"
        
        legend_html += "</div>"
        
        # Add network statistics
        stats_html = "<div style='position:absolute; bottom: 10px; left: 10px; background-color: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px;'>"
        stats_html += f"<div><b>Total nodes:</b> {len(graph.nodes)}</div>"
        stats_html += f"<div><b>Total edges:</b> {len(graph.edges)}</div>"
        stats_html += "</div>"
        
        # Write HTML with UTF-8 encoding
        html_str = net.generate_html()
        # Insert legend and stats before the closing body tag
        html_str = html_str.replace("</body>", f"{legend_html}{stats_html}</body>")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_str)

        if display:
            try:
                from IPython.display import HTML
                from IPython.display import display
                display(HTML(html_str))
            except ImportError:
                pass  # IPython not available in CLI environment
                
        print(f"Enhanced visualization saved to: {path}")

    def generate_knowledge_graph(
        self,
        paper_paths,
        paper_texts,
        nodes_per_paper=25,
        node_creation_method='baseline',
        edge_creation_method='embedding_similarity',
        threshold=0.6,
        output_path=None,
        display=False,
        custom_colors=None,
        advanced_visualization=True,
    ):
        """
        Generate a knowledge graph from multiple research papers
        Compatible with existing interface but with enhanced functionality
        """

        # 1. Load papers
        print(f"Loading {len(paper_paths)} papers...")
        papers_dict = paper_texts
        if not papers_dict:
            print("No papers were successfully loaded. Exiting.")
            return None
        
        # 2. Summarize all papers once for both node and edge creation
        print("Summarizing papers for improved processing...")
        summarized_papers = {}
        for paper_path, paper_text in papers_dict.items():
            summary = self.summarize_paper(paper_text, paper_path)
            summarized_papers[paper_path] = summary
        
        # Define colors for each paper if not provided
        if custom_colors is None:
            # Use distinctive colors rather than random ones
            distinctive_colors = [
                "#4285F4",  # Google Blue
                "#EA4335",  # Google Red
                "#FBBC05",  # Google Yellow
                "#34A853",  # Google Green
                "#FF9900",  # Amazon Orange
                "#146EB4",  # Walmart Blue
                "#00A1E0",  # Salesforce Blue
                "#0F9D58",  # Android Green
                "#AB47BC",  # Purple
                "#00BCD4",  # Cyan
                "#FF5722",  # Deep Orange
                "#795548"   # Brown
            ]
            paper_colors = {}
            for i, path in enumerate(paper_paths):
                paper_colors[path] = distinctive_colors[i % len(distinctive_colors)]
        else:
            paper_colors = custom_colors

        # 3. Create nodes for each paper using summaries
        print(f"Creating nodes using enhanced method ({nodes_per_paper} nodes per paper)...")
        combined_nodes = []
        
        # Create nodes for each paper using summarized text
        for paper_path, paper_summary in summarized_papers.items():
            paper_nodes = self.create_nodes(paper_summary, nodes_per_paper, paper_path)
            combined_nodes.extend(paper_nodes)
            print(f"  Created {len(paper_nodes)} nodes from {paper_path}")
        
        # 4. Create edges using summaries
        print(f"Creating edges using enhanced method...")
        edges = self.create_edges(
            nodes_with_source=combined_nodes,
            summarized_papers=summarized_papers,
        ) 
        print(f"  Created {len(edges)} edges")

        # 5. Ensure graph connectivity
        print("Ensuring graph connectivity...")
        edges = self.ensure_graph_connectivity(combined_nodes, edges)
        
        # 6. Create graph
        print("Building and visualizing the knowledge graph...")
        knowledge_graph = self.create_graph(combined_nodes, edges, paper_colors)
        
        # Export knowledge graph data to .dashkg.json format
        dashkg_output_path = output_path.replace('.html', '.dashkg.json') if output_path else f"kg_enhanced_{len(knowledge_graph.nodes)}_nodes.dashkg.json"
        self.export_graph_to_dashkg_json(knowledge_graph, dashkg_output_path)
        
        # 7. Generate descriptive output path if not provided
        total_nodes = len(knowledge_graph.nodes)
        if output_path is None:
            output_path = f"kg_enhanced_{total_nodes}_nodes.html"
        
        # 8. Export graph to HTML - use advanced visualization
        self.advanced_graph_to_html(knowledge_graph, output_path, display)
        
        print(f"Knowledge graph generated and saved to {output_path}")
        print(f"Total nodes: {total_nodes}")
        print(f"Total edges: {len(edges)}")
        
        return knowledge_graph


# Compatibility functions for existing nano-graphrag integration
def create_enhanced_kg_instance() -> EnhancedKG:
    """Create enhanced KG instance for use in nano-graphrag"""
    return EnhancedKG()


# Wrapper functions to maintain compatibility with existing interface
def gemini_create_nodes(paper_text: str, node_limit: int, paper_source: str) -> list:
    """Compatibility wrapper for existing gemini_create_nodes function"""
    kg_instance = create_enhanced_kg_instance()
    return kg_instance.create_nodes(paper_text, node_limit, paper_source)


def create_edges_by_gemini(nodes_with_source, papers_dict):
    """Compatibility wrapper for existing create_edges_by_gemini function"""
    kg_instance = create_enhanced_kg_instance()
    return kg_instance.create_edges(nodes_with_source, papers_dict)