import gym
from old_src.build_curriculum import *

env_dict = {}
for idx, env_str in enumerate(env_list):
    env = gym.make(env_str)
    env_dict[env_str] = {
        # how big the enviroment is
        "env": env,
        "env_name": env_str,
        "n_grid": env.height * env.width,
        # how far the agent can see (usually all envs are the same)
        "agent_view_size": env.agent_view_size,
        # how many rooms in the env
        "rooms": len(env.rooms) if hasattr(env, "rooms") else 1,
        # how many objects in the env (door1, door2, key1, key2, wall, ...)
        "n_objects": len(set([obj.type + obj.color for obj in env.grid.grid if obj])),
        "objects": set([obj.type for obj in env.grid.grid if obj]),
        "potential": 1,
        "id": idx,

    }
OBJECTS = ["ball", "door", "wall", "box", "goal", "key"]


#TODO: build graph make use of the above information
def generate(sources, target):
    '''
    Input:
        sources: list of env
        target: env
    Output:
    '''
<<<<<<< Updated upstream
    # pruned = prune(sources, target)
    groups = generateGroups(sources)
    edges = inter_group_transfer(groups)

=======
    #pruned = prune(sources, target)
    generateGroups(sources)
>>>>>>> Stashed changes

def prune(sources, target):
    potentials = calculate_potentials(sources, target)
    return #the remaining source envs


def calculate_potentials(sources, target):
    potentials = []
    for source in sources:
        potential = calculate(source, target)
        potentials.append([source, potential])

def calculate(source, target):
    return 0


def extract_feature(env):
    feature_1 = [env["n_grid"], env["agent_view_size"], env["rooms"], env["n_objects"], env["potential"]]
    feature_2 = [1 if obj in env["objects"] else 0 for obj in OBJECTS]
    feature = feature_1 + feature_2
    feature = np.array(feature)
    return feature


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
    edges = np.zeros((38, 38))
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
                edges[source_id, target_id] = 1
    return edges

def exp_group(group,i):
    assert i<len(group)
    return [group[j] for j in range(i, len(group))]




def intra_group_edges(group, sigma):
    ''''
    input:
    group:list
    sigma:float
    -----------------
    return:
    adj_matrix:np.array(len(gourp,group))
    '''
    l_group=len(group)
    adj_matrix = np.zeros((l_group,l_group))
    for i in range(l_group-1):
        potentials = calculate_potentials(exp_group(group,i), group[i])
        max_idx= np.argmax(potentials)
        if potentials[max_idx]>=sigma:
            adj_matrix[max_idx+i,i] = 1
    return adj_matrix







# def inter_group_edges(group_s, group_t,sigma):
#     '''
#     input:
#     group_s:group, source
#     group_t:group, target
#     sigma:float
#     ------------------------------------
#     output:
#     adj_matrix:np.array((len(group_s), len(group_t)))
#     '''
#
#     l_group_t = len(group_t)
#     l_group_s = len(group_s)
#     adj_matrix = np.zeros((l_group_s, l_group_t))
#     for i in range(l_group_t):
#         potentials = calculate_potentials(group_s, group_t[i])
#         max_idx = np.argmax(potentials)
#         if potentials[max_idx]>=sigma:
#             adj_matrix[max_idx,i] = 1












if __name__ == '__main__':
    target = env_dict['MiniGrid-ObstructedMaze-Full-v0']
    sources = [env_dict[env_name] for env_name in env_list if env_name != "MiniGrid-ObstructedMaze-Full-v0"]
    generate(sources, target)
    pass