import gym_novel_gridworlds
import gym

no_of_environmets = 4
width_array = [11, 8, 11, 11]
height_array = [11, 8, 11, 11]
no_trees_array = [0, 1, 3, 4]
no_rocks_array = [0, 1, 2, 2]
no_table_array = [1, 0, 1, 1]
tent_area_array = [1, 0, 1, 1]
starting_trees_array = [0, 0, 0, 0]
starting_rocks_array = [0, 0, 0, 0]
starting_pogo_sticks_array = [1, 0, 0, 0]
type_of_env_array = [3, 1, 2, 2]
no_fires_array = [1, 0, 3, 1]
env_id = "NovelGridworld-v0"


def get_envs():
    envs = []
    for i_env, (width, height, no_tree, no_rock, no_table, tent_area, start_tree, start_rock,
                no_fire, starting_pogo_sticks, type_of_env) in \
        enumerate(zip(width_array, height_array, no_trees_array,
                      no_rocks_array, no_table_array, tent_area_array, starting_trees_array, starting_rocks_array,
                      no_fires_array, starting_pogo_sticks_array,type_of_env_array)):
        env = gym.make(env_id,
                       map_width=width,
                       map_height=height,
                       items_quantity={'tree': no_tree, 'rock': no_rock, 'fire': no_fire, 'crafting_table': no_table, 'pogo_stick': 0,
                                       'tent': 0, 'tent_area': tent_area},
                       initial_inventory={'wall': 0, 'tree': start_tree, 'rock': start_rock, 'fire': 0, 'crafting_table': 0,
                                          'pogo_stick': starting_pogo_sticks, 'tent': 0, 'tent_area': 0},
                       no_fire=no_fire,
                       goal_env=type_of_env,
                       is_final=i_env==(no_of_environmets-1))
        envs.append(env)
    return envs

if __name__ == '__main__':
    get_envs()