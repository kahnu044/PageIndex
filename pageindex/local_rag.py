# local_rag.py — Pure-local RAG using a pre-built PageIndex tree JSON
# No PageIndex cloud API calls needed. Only an LLM API key is required.
#
# Usage:
#   python -m pageindex.local_rag --tree results/agentic-ai_structure.json --query "What are the conclusions?"
#
# Env vars for LLM (handled by pageindex.utils):
#   LLM_PROVIDER  = "openai" (default) or "anthropic"
#   LLM_API_KEY   = your provider API key
#   LLM_MODEL     = optional model override

import argparse
import json
import asyncio
import pageindex.utils as utils

# -------------------------
# Default paths
# -------------------------
DEFAULT_TREE_PATH = "results/agentic-ai_structure.json"
DEFAULT_QUERY = "What are the conclusions in this document?"


# -------------------------
# Async LLM helper – delegates to shared provider-agnostic util
# -------------------------
async def call_llm(prompt: str, model: str = None) -> str:
    """
    Calls the LLM using the shared utils.ChatGPT_API_async function.
    Supports both OpenAI and Anthropic based on the LLM_PROVIDER env var.
    """
    model = model or utils._get_default_model()
    result = await utils.ChatGPT_API_async(model, prompt)
    return result.strip() if result else result


# -------------------------
# Build a flat node_id -> node mapping from a nested tree
# -------------------------
def build_node_map(tree):
    """Recursively flatten the tree into a dict keyed by node_id."""
    node_map = {}

    def _walk(nodes):
        for node in nodes:
            nid = node.get("node_id")
            if nid:
                node_map[nid] = node
            if node.get("nodes"):
                _walk(node["nodes"])

    if isinstance(tree, list):
        _walk(tree)
    elif isinstance(tree, dict):
        _walk([tree])
    return node_map


# -------------------------
# Main async flow
# -------------------------
async def main(tree_path: str, query: str):
    # Step 1: Load the tree JSON from disk
    print(f"Loading tree from: {tree_path}")
    with open(tree_path, "r", encoding="utf-8") as f:
        tree_data = json.load(f)

    # Handle both raw list and {"structure": [...]} wrapper
    if isinstance(tree_data, dict) and "structure" in tree_data:
        tree = tree_data["structure"]
        doc_name = tree_data.get("doc_name", tree_path)
    elif isinstance(tree_data, list):
        tree = tree_data
        doc_name = tree_path
    else:
        raise ValueError(f"Unexpected tree JSON format in {tree_path}")

    print(f"Document: {doc_name}")
    print(f"Top-level nodes: {len(tree)}")

    # Build a flat node map for easy lookup
    node_map = build_node_map(tree)
    print(f"Total nodes (flat): {len(node_map)}")

    # Step 2: LLM-directed tree search
    # Remove text/summary bodies to keep the search prompt small
    tree_without_text = utils.remove_fields(tree, fields=["text", "summary"])
    search_prompt = f"""
        You are given a question and a tree structure of a document.
        Each node contains a node_id and a title.
        Your task is to find all nodes that are likely to contain the answer to the question.

        Question: {query}

        Document tree structure:
        {json.dumps(tree_without_text, indent=2)}

        Please reply in the following JSON format:
        {{
            "thinking": "<Your reasoning about which nodes are relevant>",
            "node_list": ["node_id_1", "node_id_2"]
        }}
        Return ONLY the JSON. Do not output anything else.
        """
    print("\nAsking LLM to search the tree for relevant nodes...")
    tree_search_result_text = await call_llm(search_prompt)
    tree_search_result = json.loads(tree_search_result_text)
    print("\n=== LLM reasoning (condensed) ===")
    print(tree_search_result.get("thinking", "")[:1000], "\n")
    print("Node IDs returned:", tree_search_result.get("node_list"))

    # Step 3: Gather context from the selected nodes
    retrieved_node_ids = tree_search_result.get("node_list", [])
    retrieved_texts = []
    for nid in retrieved_node_ids:
        node = node_map.get(nid)
        if not node:
            continue
        # Prefer 'text' if available, fall back to 'summary'
        node_content = node.get("text") or node.get("summary") or ""
        if isinstance(node_content, list):
            node_content = "\n\n".join(node_content)
        retrieved_texts.append(
            f"--- Node {nid}: {node.get('title', 'Untitled')} ---\n{node_content}"
        )
    combined_context = "\n\n".join(retrieved_texts) or "No context retrieved."

    # Step 4: Answer generation from retrieved context
    answer_prompt = f"""
Answer the question based only on the context below.

Question: {query}

Context:
{combined_context}

Provide a concise, grounded answer and include the node IDs you used.
"""
    print(f"\nGenerating final answer with prompt: {answer_prompt}")
    final_answer = await call_llm(answer_prompt)
    print("\n=== Final answer ===\n")
    print(final_answer)


def cli():
    parser = argparse.ArgumentParser(
        description="Local RAG: query a PageIndex tree JSON with an LLM (no cloud API needed)"
    )
    parser.add_argument(
        "--tree",
        type=str,
        default=DEFAULT_TREE_PATH,
        help=f"Path to the *_structure.json tree file (default: {DEFAULT_TREE_PATH})",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Question to ask about the document",
    )
    args = parser.parse_args()
    asyncio.run(main(args.tree, args.query))


if __name__ == "__main__":
    cli()