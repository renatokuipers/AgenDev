import time

import random

class Player:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.hunger = 100
        self.thirst = 100
        self.inventory = []
        self.sanity = 100
        self.root = tk.Tk()
        self.root.title("Horror Game")
        self.label = tk.Label(self.root, text="Welcome to the horror game!")
        self.label.pack()
        self.entry = tk.Entry(self.root)
        self.entry.pack()
        self.button = tk.Button(self.root, text="Submit", command=self.submit_action)
        self.button.pack()
        self.text_box = tk.Text(self.root)
        self.text_box.pack()

    def is_alive(self):
        return self.health > 0 and self.sanity > 0 and self.hunger > 0 and self.thirst > 0

    def submit_action(self):
        action = self.entry.get()
        self.entry.delete(0, tk.END)
        # rest of the game logic will go here

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
            "entrance": {"description": "You are at the entrance of an abandoned asylum. You've been searching for your missing sister for weeks, and you finally got a tip that she might be here.", "north": "hallway", "south": "garden"},
            "hallway": {"description": "You are in a long, dark hallway. You can hear the sound of footsteps echoing in the distance.", "south": "entrance", "north": "room"},
            "garden": {"description": "You are in a overgrown garden. You can see a figure in the distance, but it's too far away to make out any features.", "north": "entrance"},
            "room": {"description": "You are in a room that looks like it was once a patient's room. There's a bed in the corner, and a small table with a journal on it.", "south": "hallway"}
        }
        self.story_progress = 0
        self.player.text_box.insert(tk.END, self.rooms[self.current_room]["description"])
        self.player.text_box.insert(tk.END, "\nAvailable actions:")
        available_items = [item for item in self.rooms[self.current_room] if item not in ["description", "north", "south"]]
        if available_items:
            self.player.text_box.insert(tk.END, "\n  take item")
        if self.player.inventory:
            self.player.text_box.insert(tk.END, "\n  use item")
        self.player.text_box.insert(tk.END, "\n  eat")
        self.player.text_box.insert(tk.END, "\n  drink")
        self.player.root.update()

    def play(self):
        print("Welcome to the horror game!")
        while self.player.is_alive():
            print("\n" + self.rooms[self.current_room]["description"])
            print("Available actions:")
            print("  go north")
            print("  go south")
            available_items = [item for item in self.rooms[self.current_room] if item not in ["description", "north", "south"]]
            if available_items:
                print("  take item")
            if self.player.inventory:
                print("  use item")
            print("  eat")
            print("  drink")
            action = input("What do you want to do? ")
            if action.lower() == "go north" and "north" in self.rooms[self.current_room]:
                self.current_room = self.rooms[self.current_room]["north"]
                self.player.hunger -= 10
                self.player.thirst -= 10
            elif action.lower() == "go south" and "south" in self.rooms[self.current_room]:
                self.current_room = self.rooms[self.current_room]["south"]
                self.player.hunger -= 10
                self.player.thirst -= 10
            elif action.lower() == "take item":
                if self.current_room == "room" and self.story_progress == 0:
                    self.player.inventory.append("journal")
                    print("You took the journal.")
                    self.story_progress += 1
                else:
                    print("There's nothing to take.")
            elif action.lower() == "use item":
                if "journal" in self.player.inventory:
                    print("You read the journal. It seems your sister was a patient here, and she was being treated for a rare mental illness.")
                    self.story_progress += 1
                else:
                    print("You don't have anything to use.")
            elif action.lower() == "eat":
                if "food" in self.player.inventory:
                    self.player.hunger += 50
                    self.player.inventory.remove("food")
                    print("You ate some food.")
                else:
                    print("You don't have any food.")
            elif action.lower() == "drink":
                if "water" in self.player.inventory:
                    self.player.thirst += 50
                    self.player.inventory.remove("water")
                    print("You drank some water.")
                else:
                    print("You don't have any water.")
            else:
                print("Invalid action.")
                self.player.lose_sanity(10)
            if random.random() < 0.1:
                self.player.take_damage(10)
                print("You were attacked by a monster!")
            if self.player.hunger <= 0:
                self.player.take_damage(10)
                print("You're starving!")
            if self.player.thirst <= 0:
                self.player.take_damage(10)
                print("You're dehydrated!")
            time.sleep(1)
        print("Game over.")
        if self.player.health <= 0:
            print("You died.")
        elif self.player.sanity <= 0:
            print("You went insane.")
        elif self.story_progress == 2:
            print("You found your sister! She's alive and well. Congratulations, you won!")

game = Game()
game.play()
