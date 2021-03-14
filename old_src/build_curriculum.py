import gym
from sklearn.cluster import KMeans
from old_src.config import env_list
import numpy as np

pos_inf = float('inf')     # positive infinity
neg_inf = float('-inf')    # negative infinity
not_a_num = float('nan')   # NaN ("not a number")
MINIMUM_POTENTIAL_THRESHOLD = 0.01
OBJECTS = ["ball", "door", "wall", "box", "goal", "key"]

env_dict = {}
for idx, env_str in enumerate(env_list):
    env = gym.make(env_str)
    env_dict[env_str] = {
        # how big the enviroment is
        "env_name": env_str,
        "n_grid": env.height * env.width,
        # how far the agent can see (usually all envs are the same)
        "agent_view_size": env.agent_view_size,
        # how many rooms in the env
        "rooms": len(env.rooms) if hasattr(env, "rooms") else 1,
        # how many objects in the env (door1, door2, key1, key2, wall, ...)
        "n_objects": len(set([obj.type + obj.color for obj in env.grid.grid if obj])),
        "objects": set([obj.type for obj in env.grid.grid if obj]),
        "potential": 1, #place holder
        "id": idx
    }

def extract_feature(env):
    feature_1 = [env["n_grid"], env["agent_view_size"], env["rooms"], env["n_objects"], env["potential"]]
    feature_2 = [1 if obj in env["objects"] else 0 for obj in OBJECTS]
    feature = feature_1 + feature_2
    feature = np.array(feature)
    return feature


#TODO: build graph make use of the above information
def generate(sources, target):
    """
    Input:
        sources: list of env
        target: env
    Output:
    """
    pruned = prune(sources, target)
    groups = generateGroups(pruned)
    global_adjs = np.zeros((40, 40))

    for _, group in groups.items():
        global_adjs += intra_group_edges(group, 0)
    global_adjs += inter_group_transfer(groups)
    return global_adjs


def prune(sources, target):
    potentials = calculate_potentials(sources, target)
    pruned = []
    for task, potential in potentials:
        if potential > MINIMUM_POTENTIAL_THRESHOLD:
            pruned.append(task)


    if len(pruned) < 1:
        raise ValueError("Not enough useful source maps!")

    return pruned

def calculate_potentials(sources, target):
    potentials = []
    for source in sources:
        potential = calculate(source, target)
        source["potential"] = potential
        potentials.append([source, potential])
    return sorted(potentials, key=lambda x:x[1], reverse=True)


def get_volume(vector):
    product_result = 1
    for each in vector:
        if each != 0:
            product_result *= each
    return product_result


def elementwise_min(first, second):
    assert len(first) == len(second)
    results = []
    for i in range(len(first)):
        results.append(min(first[i], second[i]))
    return results


def applicability(source_features, target_features):
    """
         * Estimates the degree to which the source task can inform the target task.
         *
         * @param source
         * @param target
         * @return
    """
    elementwise_min_vector = elementwise_min(source_features, target_features)
    volume = get_volume(elementwise_min_vector)
    return volume


def cost(source_features, target_features):
    volume_target = get_volume((target_features))
    volume_source = get_volume(source_features)
    return volume_target - volume_source


def calculate(source, target):
    """"
     * Calculate the transfer potential
     *
     * @param source
     * @param target
     * @return
     """
    applicability_weight = 1.0
    source_features = extract_feature(source)
    target_features = extract_feature(target)
    a = applicability(source_features, target_features)
    c = cost(source_features, target_features)

    if c == 0.0:
        return 0

    return applicability_weight * a / c


def generateGroups(sources):
    # sources: list of env
    X = [extract_feature(source) for source in sources]
    X = np.array(X)
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0)+1)
    clusters = KMeans(n_clusters=6).fit_predict(X)
    groups = {}
    for idx in range(len(clusters)):
        if clusters[idx] in groups:
            groups[clusters[idx]].append(sources[idx])
        else:
            groups[clusters[idx]] = [sources[idx]]

    return groups


def is_subset(group1, group2):
    objects1 = set(sum([list(env["objects"]) for env in group1], []))
    objects2 = set(sum([list(env["objects"]) for env in group2], []))
    return objects1.intersection(objects2)==objects2



def inter_group_transfer(groups):
    edges = np.zeros((40, 40))
    for key, group in groups.items():
        subset_groups = []
        for target_key, target_group in groups.items():
            if key != target_key:
                if is_subset(group, target_group):
                    subset_groups.append(target_group)
        for subset_group in subset_groups:
            for source_task in group:
                potentials = []
                for target_task in subset_group:
                    potentials.append(calculate(source_task, target_task))
                best_transfer_id = np.argmax(potentials)
                target_id = subset_group[best_transfer_id]["id"]
                source_id = source_task["id"]
                if potentials[best_transfer_id] > 0.2:
                    edges[source_id, target_id] =1# potentials[best_transfer_id]
    return edges

def intra_group_edges(group, sigma):
    ''''
    input:
    group:list
    sigma:float
    -----------------
    return:
    adj_matrix:np.array(len(gourp,group))
    '''
    # l_group = len(group)
    adj_matrix = np.zeros((40, 40))
    for i in range(len(group)):
        potentials = calculate_potentials(exp_group(group, i), group[i])
        max_idx = np.argmax([potential[1] for potential in potentials])

        if potentials[max_idx][1] > 0.2:
            target_id = group[i]["id"]
            source_id = potentials[max_idx][0]["id"]
            adj_matrix[source_id, target_id] = 1# potentials[max_idx][1]
    return adj_matrix

def exp_group(group,i):
    # assert i < len(group)
    return [group[j] for j in range(0, len(group)) if i != j]


if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph  import graphviz_layout
    target = env_dict['MiniGrid-ObstructedMaze-Full-v0']
    sources = [env_dict[env_name] for env_name in env_list if env_name != "MiniGrid-ObstructedMaze-Full-v0"]
    adj = generate(sources, target)
    G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
    pos = graphviz_layout(G, prog='dot')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.draw()
    plt.show()
    list(reversed(list(nx.algorithms.dag.topological_sort(G))))