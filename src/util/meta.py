
class Param:
    @staticmethod
    def trials_name(tx):
        trials_list = {"Combination": ["Combination", "Combo", "Comb"],
                       "Define": ["Define"],
                       "Free": ["Free"],
                       "Straight": ["Straight", "Str"],
                       "Obstacle": ["Obstacle", "Osb", 'obst'],
                       "Reactive": ["Reactive", 'Rea'],
                       "Stairs": ["Stairs", "Stair"],
                       "Static": ['Stat', "Cal"],
                       "Test": ['test']
                       }

        for s in trials_list:
            for c in trials_list[s]:
                if c.lower() in tx.lower():
                    return s
        return None

    @staticmethod
    def condition(tx):
        con_list = {
            "Normal": ["Norm"],
            "AR": ["AR"],
            "VR": ["VR"]
        }

        for s in con_list:
            for c in con_list[s]:
                if c.lower() in tx.lower():
                    return s
        return "Normal"

    @staticmethod
    def trial_id(tx):
        c = tx.strip()[-1]
        try:
            a = int(c)
            return "T{0:01d}".format(a)
        except ValueError:
            return "NA"