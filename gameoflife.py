import time

class Player:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.inventory = []

    def is_alive(self):
        return self.health > 0

class Game:
    def __init__(self):
        self.player = Player("Protagonist")
        self.current_room = "entrance"

    def play(self):
        print("Welcome to the horror game!")
        while self.player.is_alive():
            print("\nYou are in the", self.current_room)
            print("Available actions:")
            print("  go north")
            print("  go south")
            print("  take key")
            print("  use key")
            action = input("What do you want to do? ")
            if action.lower() == "go north":
                self.current_room = "hallway"
            elif action.lower() == "go south":
                self.current_room = "garden"
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
            time.sleep(1)
        print("Game over.")

game = Game()
game.play()
