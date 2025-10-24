def get_descriptors(ds):
    if ds =='bierling2025':
        return ['intensity','pleasantness','familiar','edible', 'warm','sour', 'cold','sweet','fruit','spices','bakery','garlic', 'fish', 
                    'burnt', 'decayed', 'grass', 'wood', 'chemical','flower', 'musky', 'sweaty', 'ammonia']
    elif ds == 'keller2016':
        # return['intensive','pleasant','familiar','edible','bakery','sweet','fruit','fish','garlic','spices','cold','sour',
        return['intensity','pleasantness','familiarity','edible','bakery','sweet','fruit','fish','garlic','spices','cold','sour',
               'burnt','acid','warm','musky','sweaty','ammonia','decayed','wood','grass','flower','chemical']
    elif ds== 'sagar2023_v1':
        pass
    elif ds == 'sagar2023_v2':
        pass
    elif ds == 'sagar2023':
        return [ 'intensity', 'pleasantness', 'fishy', 'burnt', 'sour', 'decayed', 'musky',
    'fruity', 'sweaty', 'cool', 'floral', 'sweet', 'warm', 'bakery', 'spicy']
    elif ds == 'leffingwell':
        return [
    "alcoholic","aldehydic","alliaceous","almond","animal","anisic","apple","apricot","aromatic","balsamic",
    "banana","beefy","berry","black currant","brandy","bread","brothy","burnt","buttery","cabbage","camphoreous",
    "caramellic","catty","chamomile","cheesy","cherry","chicken","chocolate","cinnamon","citrus","cocoa","coconut",
    "coffee","cognac","coumarinic","creamy","cucumber","dairy","dry","earthy","ethereal","fatty","fermented","fishy",
    "floral","fresh","fruity","garlic","gasoline","grape","grapefruit","grassy","green","hay","hazelnut","herbal",
    "honey","horseradish","jasmine","ketonic","leafy","leathery","lemon","malty","meaty","medicinal","melon","metallic",
    "milky","mint","mushroom","musk","musty","nutty","odorless","oily","onion","orange","orris","peach","pear",
    "phenolic","pine","pineapple","plum","popcorn","potato","pungent","radish","ripe","roasted","rose","rum","savory",
    "sharp","smoky","solvent","sour","spicy","strawberry","sulfurous","sweet","tea","tobacco","tomato","tropical",
    "vanilla","vegetable","violet","warm","waxy","winey","woody"
]
    else:
        raise ValueError("Unsupported dataset: {}".format(ds))
    
    

def get_descriptors2(ds):
    if ds =='bierling2025':
        return ['intensity','pleasantness','familiar','edible', 'warm','sour', 'cold','sweet','fruit','spices','bakery','garlic', 'fish', 
                    'burnt', 'decayed', 'grass', 'wood', 'chemical','flower', 'musky', 'sweaty', 'ammonia']
    elif ds == 'keller2016':
        return['intensive','pleasant','familiar','edible','bakery','sweet','fruit','fish','garlic','spices','cold','sour',
        # return['intensity','pleasantness','familiarity','edible','bakery','sweet','fruit','fish','garlic','spices','cold','sour',
               'burnt','acid','warm','musky','sweaty','ammonia','decayed','wood','grass','flower','chemical']
    elif ds== 'sagar2023_v1':
        pass
    elif ds == 'sagar2023_v2':
        pass
    elif ds == 'sagar2023':
        return [ 'intensity', 'pleasantness', 'fishy', 'burnt', 'sour', 'decayed', 'musky',
    'fruity', 'sweaty', 'cool', 'floral', 'sweet', 'warm', 'bakery', 'spicy']
    elif ds == 'leffingwell':
        return [
    "alcoholic","aldehydic","alliaceous","almond","animal","anisic","apple","apricot","aromatic","balsamic",
    "banana","beefy","berry","black currant","brandy","bread","brothy","burnt","buttery","cabbage","camphoreous",
    "caramellic","catty","chamomile","cheesy","cherry","chicken","chocolate","cinnamon","citrus","cocoa","coconut",
    "coffee","cognac","coumarinic","creamy","cucumber","dairy","dry","earthy","ethereal","fatty","fermented","fishy",
    "floral","fresh","fruity","garlic","gasoline","grape","grapefruit","grassy","green","hay","hazelnut","herbal",
    "honey","horseradish","jasmine","ketonic","leafy","leathery","lemon","malty","meaty","medicinal","melon","metallic",
    "milky","mint","mushroom","musk","musty","nutty","odorless","oily","onion","orange","orris","peach","pear",
    "phenolic","pine","pineapple","plum","popcorn","potato","pungent","radish","ripe","roasted","rose","rum","savory",
    "sharp","smoky","solvent","sour","spicy","strawberry","sulfurous","sweet","tea","tobacco","tomato","tropical",
    "vanilla","vegetable","violet","warm","waxy","winey","woody"
]
    else:
        raise ValueError("Unsupported dataset: {}".format(ds))
    
    