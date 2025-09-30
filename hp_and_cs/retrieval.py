from hypergraph_propagation import propagate
from utils.retrieval_component import connect_nodup
from community_selection import extract_sub_graph,calculate_entropy,match_one_pair_delg,find_dominant


def rank_with_hp_cs(hp_qranks, query_index,
                    COMMUNITY_SELECTION=1,
                    HYPERGRAPH_PROPAGATION=True):
    """
    Runs first retrieval, optional community selection, and optional hypergraph propagation.

    Args:
        hp_qranks (np.ndarray): Ranking matrix where each column corresponds to a query's ranked images.
        query_index (int): Index of the current query.
        q (str): Query image name or identifier.
        COMMUNITY_SELECTION (int or bool): 
            0 or False = skip, 
            1 = only calculate uncertainty, 
            2 = also apply dominant selection based on inlier threshold.
        HYPERGRAPH_PROPAGATION (bool): Whether to apply hypergraph propagation.

    Returns:
        final_ranks (list): Ranked list of image identifiers.
        dominant_image (str): Dominant image used as the root.
    """
    # First search
    if hp_qranks.ndim == 1:
        first_search = list(hp_qranks)  # 1D: already ranks for a single query
    else:
        first_search = list(hp_qranks[:, query_index])  # 2D: multiple queries

    dominant_image = first_search[0]

    # Community selection
    if COMMUNITY_SELECTION:
        Gs = extract_sub_graph(first_search, 20)
        uncertainty = calculate_entropy(Gs)

        if COMMUNITY_SELECTION == 2 and uncertainty > 1:
            inlier, size = match_one_pair_delg(query_index, first_search[0])
            if inlier < 20:
                Gs = extract_sub_graph(first_search, 100)
                dominant_image = find_dominant(Gs, first_search, query_index)
        else:
            print(f"Uncertainty: {uncertainty}")

    # Hypergraph propagation
    if HYPERGRAPH_PROPAGATION:
        rank_list = propagate([dominant_image])
        final_ranks = connect_nodup(rank_list, first_search)
    else:
        final_ranks = first_search

    return final_ranks, dominant_image