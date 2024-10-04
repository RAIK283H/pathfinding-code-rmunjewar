import math
import pyglet
import global_game_data
import relative_display_functions
import graph_data
import config_data


class Player:

    def __init__(self, player_config_data, player_index, batch, group):
        self.speed = relative_display_functions.get_absolute_speed()
        self.current_objective = 0
        self.player_index = player_index
        self.absolute_x = graph_data.graph_data[global_game_data.current_graph_index][0][0][0]
        self.absolute_y = graph_data.graph_data[global_game_data.current_graph_index][0][0][1]
        self.player_image = pyglet.resource.image(player_config_data[1])
        self.sprite = pyglet.sprite.Sprite(img=self.player_image, x=0, y=0, batch=batch, group=group)
        self.player_config_data = player_config_data
        self.distance_traveled = 0
        self.nodes_visited = set()

    def update_location(self, x, y):
        self.sprite.update(relative_display_functions.get_relative_graph_x(x) - self.sprite.width / 2,
                           relative_display_functions.get_relative_graph_y(y) - self.sprite.height / 2)

    def reset_player(self):
        self.current_objective = 0
        self.absolute_x = graph_data.graph_data[global_game_data.current_graph_index][0][0][0]
        self.absolute_y = graph_data.graph_data[global_game_data.current_graph_index][0][0][1]
        self.distance_traveled = 0
        self.nodes_visited.clear()

    def update(self, dt):
        last_absolute_x = self.absolute_x
        last_absolute_y = self.absolute_y
        # Make sure one player is always running
        if global_game_data.current_player_index < 0 or global_game_data.current_player_index >= len(config_data.player_data):
            global_game_data.current_player_index = 0
            # Reset all values of players
            for player in global_game_data.player_objects:
                player.reset_player()
            return

        # Move to next player if current objective is out of bounds
        if global_game_data.current_player_index == self.player_index:
            if self.current_objective >= len(global_game_data.graph_paths[self.player_index]):
                self.current_objective = 0
                global_game_data.current_player_index = global_game_data.current_player_index + 1

        # Move player under normal circumstances
        if self.current_objective >= 0 and global_game_data.current_player_index == self.player_index:
            current_node = global_game_data.graph_paths[self.player_index][self.current_objective]
            self.nodes_visited.add(current_node)
            target_x = graph_data.graph_data[global_game_data.current_graph_index][
                global_game_data.graph_paths[self.player_index][self.current_objective]][0][0]
            target_y = graph_data.graph_data[global_game_data.current_graph_index][
                global_game_data.graph_paths[self.player_index][self.current_objective]][0][1]
            difference_in_x = target_x - self.absolute_x
            difference_in_y = target_y - self.absolute_y
            difference = math.sqrt(pow(difference_in_x, 2) + pow(difference_in_y, 2))
            change_in_x = 0
            change_in_y = 0
            if difference > 0:
                change_in_x = difference_in_x / difference * self.speed * dt
                change_in_y = difference_in_y / difference * self.speed * dt
            # If statements make sure player does not overshoot target
            if self.absolute_x == target_x or self.absolute_x < target_x < self.absolute_x + change_in_x or \
                    self.absolute_x + change_in_x < target_x < self.absolute_x:
                self.absolute_x = target_x
            else:
                self.absolute_x = self.absolute_x + change_in_x

            if self.absolute_y == target_y or self.absolute_y < target_y < self.absolute_y + change_in_y or \
                    self.absolute_y + change_in_y < target_y < self.absolute_y:
                self.absolute_y = target_y
            else:
                self.absolute_y = self.absolute_y + change_in_y
            # Go to next object when target is reached
            if self.absolute_x == target_x and self.absolute_y == target_y:
                self.current_objective += 1

        self.distance_traveled = self.distance_traveled + math.sqrt(math.pow(last_absolute_x-self.absolute_x, 2) + math.pow(last_absolute_y-self.absolute_y, 2))
        self.sprite.visible = (global_game_data.current_player_index == self.player_index)
        self.update_location(self.absolute_x, self.absolute_y)
