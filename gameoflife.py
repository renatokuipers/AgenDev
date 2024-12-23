import time

import random

class Player:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.inventory = []
        self.sanity = 100

    def is_alive(self):
        return self.health > 0 and self.sanity > 0

    def take_damage(self, amount):
        self.health -= amount
        if self.health < 0:
            self.health = 0

    def lose_sanity(self, amount):
        self.sanity -= amount
        if self.sanity < 0:
            self.sanity = 0

class Game:
    def __init__(self):
        self.player = Player("Protagonist")
        self.current_room = "entrance"
        self.rooms = {
            "entrance": {"description": "You are at the entrance of the house.", "north": "hallway", "south": "garden"},
            "hallway": {"description": "You are in the hallway.", "south": "entrance", "north": "room"},
            "garden": {"description": "You are in the garden.", "north": "entrance"},
            "room": {"description": "You are in a room.", "south": "hallway"}
        }

    def play(self):
        print("Welcome to the horror game!")
        while self.player.is_alive():
            print("\n" + self.rooms[self.current_room]["description"])
            print("Available actions:")
            print("  go north")
            print("  go south")
            print("  take key")
            print("  use key")
            action = input("What do you want to do? ")
            if action.lower() == "go north" and "north" in self.rooms[self.current_room]:
                self.current_room = self.rooms[self.current_room]["north"]
            elif action.lower() == "go south" and "south" in self.rooms[self.current_room]:
                self.current_room = self.rooms[self.current_room]["south"]
            elif action.lower() == "take key":
                self.player.inventory.append("key")
                print("You took the key.")
            elif action.lower() == "use key":
                if "key" in self.player.inventory:
                    print("You unlocked the door.")
                    self.current_room = "room"
                else:
                    print("You don't have a key.")
            else:
                print("Invalid action.")
                self.player.lose_sanity(10)
            if random.random() < 0.1:
                self.player.take_damage(10)
                print("You were attacked by a monster!")
            time.sleep(1)
        print("Game over.")
        if self.player.health <= 0:
            print("You died.")
        elif self.player.sanity <= 0:
            print("You went insane.")

game = Game()
game.play()
