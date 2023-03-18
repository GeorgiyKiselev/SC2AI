# %%
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race 
from sc2.main import run_game  
from sc2.player import Bot, Computer 
from sc2 import maps  
from sc2.ids.unit_typeid import UnitTypeId
import random
import cv2
import math
import numpy as np
import sys
import pickle
import time


SAVE_REPLAY = True

total_steps = 10000 
steps_for_pun = np.linspace(0, 1, total_steps)
step_punishment = ((np.exp(steps_for_pun**3)/10) - 0.1)*10



class SC2BOT(BotAI): 
    async def on_step(self, iteration: int): 
        no_action = True
        while no_action:
            try:
                with open('state_rwd_action.pkl', 'rb') as f:
                    state_rwd_action = pickle.load(f)

                    if state_rwd_action['action'] is None:
                        no_action = True
                    else:
                        no_action = False
            except:
                pass


        await self.distribute_workers() 

        action = state_rwd_action['action']
        '''
        0: expand (ie: move to next spot, or build to 16 (minerals)+3 assemblers+3)
        1: build stargate (or up to one) (evenly)
        2: build voidray (evenly)
        3: send scout (evenly/random/closest to enemy?)
        4: attack (known buildings, units, then enemy base, just go in logical order.)
        5: voidray flee (back to base)
        '''

        if action == 0:
            try:
                found_something = False
                if self.supply_left < 4:
                    if self.already_pending(UnitTypeId.PYLON) == 0:
                        if self.can_afford(UnitTypeId.PYLON):
                            await self.build(UnitTypeId.PYLON, near=random.choice(self.townhalls))
                            found_something = True

                if not found_something:

                    for nexus in self.townhalls:
                        worker_count = len(self.workers.closer_than(10, nexus))
                        if worker_count < 24: 
                            if nexus.is_idle and self.can_afford(UnitTypeId.PROBE):
                                nexus.train(UnitTypeId.PROBE)
                                found_something = True

                        for geyser in self.vespene_geyser.closer_than(10, nexus):
                            if not self.can_afford(UnitTypeId.ASSIMILATOR):
                                break
                            if not self.structures(UnitTypeId.ASSIMILATOR).closer_than(2.0, geyser).exists:
                                await self.build(UnitTypeId.ASSIMILATOR, geyser)
                                found_something = True

                if not found_something:
                    if self.already_pending(UnitTypeId.NEXUS) == 0 and self.can_afford(UnitTypeId.NEXUS):
                        await self.expand_now()

            except Exception as e:
                print(e)


        elif action == 1:
            try:
                for nexus in self.townhalls:
                    if not self.structures(UnitTypeId.GATEWAY).closer_than(10, nexus).exists:
                        if self.can_afford(UnitTypeId.GATEWAY) and self.already_pending(UnitTypeId.GATEWAY) == 0:
                            await self.build(UnitTypeId.GATEWAY, near=nexus)
                        
                    if not self.structures(UnitTypeId.CYBERNETICSCORE).closer_than(10, nexus).exists:
                        if self.can_afford(UnitTypeId.CYBERNETICSCORE) and self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0:
                            await self.build(UnitTypeId.CYBERNETICSCORE, near=nexus)

                    if not self.structures(UnitTypeId.STARGATE).closer_than(10, nexus).exists:
                        if self.can_afford(UnitTypeId.STARGATE) and self.already_pending(UnitTypeId.STARGATE) == 0:
                            await self.build(UnitTypeId.STARGATE, near=nexus)

            except Exception as e:
                print(e)


        elif action == 2:
            try:
                if self.can_afford(UnitTypeId.VOIDRAY):
                    for sg in self.structures(UnitTypeId.STARGATE).ready.idle:
                        if self.can_afford(UnitTypeId.VOIDRAY):
                            sg.train(UnitTypeId.VOIDRAY)
            
            except Exception as e:
                print(e)

        elif action == 3:
            try:
                self.last_sent
            except:
                self.last_sent = 0

            if (iteration - self.last_sent) > 200:
                try:
                    if self.units(UnitTypeId.PROBE).idle.exists:
                        probe = random.choice(self.units(UnitTypeId.PROBE).idle)
                    else:
                        probe = random.choice(self.units(UnitTypeId.PROBE))
                    probe.attack(self.enemy_start_locations[0])
                    self.last_sent = iteration

                except Exception as e:
                    pass


        elif action == 4:
            try:
                for voidray in self.units(UnitTypeId.VOIDRAY).idle:
                    if self.enemy_units.closer_than(10, voidray):
                        voidray.attack(random.choice(self.enemy_units.closer_than(10, voidray)))
                    elif self.enemy_structures.closer_than(10, voidray):
                        voidray.attack(random.choice(self.enemy_structures.closer_than(10, voidray)))
                    elif self.enemy_units:
                        voidray.attack(random.choice(self.enemy_units))
                    elif self.enemy_structures:
                        voidray.attack(random.choice(self.enemy_structures))
                    elif self.enemy_start_locations:
                        voidray.attack(self.enemy_start_locations[0])
            
            except Exception as e:
                print(e)
            

        elif action == 5:
            if self.units(UnitTypeId.VOIDRAY).amount > 0:
                for vr in self.units(UnitTypeId.VOIDRAY):
                    vr.attack(self.start_location)


        map = np.zeros((self.game_info.map_size[0], self.game_info.map_size[1], 3), dtype=np.uint8)

        for mineral in self.mineral_field:
            pos = mineral.position
            c = [175, 255, 255]
            fraction = mineral.mineral_contents / 1800
            if mineral.is_visible:
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]
            else:
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [20,75,50]  


        for enemy_start_location in self.enemy_start_locations:
            pos = enemy_start_location
            c = [0, 0, 255]
            map[math.ceil(pos.y)][math.ceil(pos.x)] = c

        for enemy_unit in self.enemy_units:
            pos = enemy_unit.position
            c = [100, 0, 255]
            fraction = enemy_unit.health / enemy_unit.health_max if enemy_unit.health_max > 0 else 0.0001
            map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]


        for enemy_structure in self.enemy_structures:
            pos = enemy_structure.position
            c = [0, 100, 255]
            fraction = enemy_structure.health / enemy_structure.health_max if enemy_structure.health_max > 0 else 0.0001
            map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]

        for our_structure in self.structures:
            if our_structure.type_id == UnitTypeId.NEXUS:
                pos = our_structure.position
                c = [255, 255, 175]
                fraction = our_structure.health / our_structure.health_max if our_structure.health_max > 0 else 0.0001
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]
            
            else:
                pos = our_structure.position
                c = [0, 255, 175]
                fraction = our_structure.health / our_structure.health_max if our_structure.health_max > 0 else 0.0001
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]


        for vespene in self.vespene_geyser:
            pos = vespene.position
            c = [255, 175, 255]
            fraction = vespene.vespene_contents / 2250

            if vespene.is_visible:
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]
            else:
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [50,20,75]

        for our_unit in self.units:
            if our_unit.type_id == UnitTypeId.VOIDRAY:
                pos = our_unit.position
                c = [255, 75 , 75]
                fraction = our_unit.health / our_unit.health_max if our_unit.health_max > 0 else 0.0001
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]


            else:
                pos = our_unit.position
                c = [175, 255, 0]
                fraction = our_unit.health / our_unit.health_max if our_unit.health_max > 0 else 0.0001
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]


        cv2.imshow('map',cv2.flip(cv2.resize(map, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), 0))
        cv2.waitKey(1)

        if SAVE_REPLAY:
            cv2.imwrite(f"replays/{int(time.time())}-{iteration}.png", map)



        reward = 0

        try:
            attack_count = 0
            for voidray in self.units(UnitTypeId.VOIDRAY):
                if voidray.is_attacking and voidray.target_in_range:
                    if self.enemy_units.closer_than(8, voidray) or self.enemy_structures.closer_than(8, voidray):
                        reward += 0.015  
                        attack_count += 1

        except Exception as e:
            print("reward",e)
            reward = 0

        
        if iteration % 100 == 0:
            print(f"Iter: {iteration}. RWD: {reward}. VR: {self.units(UnitTypeId.VOIDRAY).amount}")


        data = {"state": map, "reward": reward, "action": None, "done": False}  

        with open('state_rwd_action.pkl', 'wb') as f:
            pickle.dump(data, f)

        


result = run_game(  
    maps.get("2000AtmospheresAIE"),
    [Bot(Race.Protoss, SC2BOT()), 
     Computer(Race.Zerg, Difficulty.Hard)],
    realtime=False, 
)


if str(result) == "Result.Victory":
    rwd = 500
else:
    rwd = -500

with open("results.txt","a") as f:
    f.write(f"{result}\n")


map = np.zeros((224, 224, 3), dtype=np.uint8)
observation = map
data = {"state": map, "reward": rwd, "action": None, "done": True}  
with open('state_rwd_action.pkl', 'wb') as f:
    pickle.dump(data, f)

cv2.destroyAllWindows()
cv2.waitKey(1)
time.sleep(3)
sys.exit()


