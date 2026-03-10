"""
ImageNet-S Dataset Loader
Supports ImageNet-S50, ImageNet-S300, and ImageNet-S919
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2

all_texts_50 =["goldfish", "tiger_shark", "goldfinch", "tree_frog", "kuvasz",
                  "red_fox", "Siamese_cat", "American_black_bear", "ladybug", "sulphur_butterfly",
                  "wood_rabbit", "hamster", "wild_boar", "gibbon", "African_elephant",
                  "giant_panda", "airliner", "ashcan", "ballpoint", "beach_wagon",
                  "boathouse", "bullet_train", "cellular_telephone", "chest", "clog",
                  "container_ship", "digital_watch", "dining_table", "golf_ball", "grand_piano",
                  "iron", "lab_coat", "mixing_bowl", "motor_scooter", "padlock",
                  "park_bench", "purse", "streetcar", "table_lamp", "television",
                  "toilet_seat", "umbrella", "vase", "water_bottle", "water_tower",
                  "yawl", "street_sign", "lemon", "carbonara", "agaric"]

all_texts_300=["tench","goldfish","tiger_shark","hammerhead","electric_ray",
 "ostrich","goldfinch","house_finch","indigo_bunting","kite",
 "common_newt","axolotl","tree_frog","tailed_frog","mud_turtle",
 "banded_gecko","American_chameleon","whiptail","African_chameleon","Komodo_dragon",
 "American_alligator","triceratops","thunder_snake","ringneck_snake","king_snake",
 "rock_python","horned_viper","harvestman","scorpion","garden_spider",
 "tick","African_grey","lorikeet","red-breasted_merganser","wallaby",
 "koala","jellyfish","sea_anemone","conch","fiddler_crab",
 "American_lobster","spiny_lobster","isopod","bittern", "crane",
 "limpkin","bustard","albatross","toy_terrier", "Afghan_hound",
 "bluetick","borzoi","Irish_wolfhound","whippet", "Ibizan_hound",
"Staffordshire_bullterrier","Border_terrier","Yorkshire_terrier","Lakeland_terrier", "giant_schnauzer",
"standard_schnauzer","Scotch_terrier","Lhasa","English_setter", "clumber",
"English_springer","Welsh_springer_spaniel","kuvasz","kelpie", "Doberman",
"miniature_pinscher","malamute","pug","Leonberg","Great_Pyrenees",
 "Samoyed","Brabancon_griffon","Cardigan","coyote","red_fox",
 "kit_fox","grey_fox","Persian_cat","Siamese_cat","cougar",
"lynx","tiger","American_black_bear","sloth_bear","ladybug",
"leaf_beetle","weevil","bee","cicada","leafhopper",
"damselfly","ringlet","cabbage_butterfly","sulphur_butterfly","sea_cucumber",
"wood_rabbit","hare","hamster","wild_boar","hippopotamus",
"bighorn","ibex","badger","three-toed_sloth","orangutan",
"gibbon","colobus","spider_monkey","squirrel_monkey","Madagascar_cat",
"Indian_elephant","African_elephant","giant_panda","barracouta","eel",
"coho","academic_gown","accordion","airliner","ambulance",
"analog_clock","ashcan","backpack","balloon","ballpoint",
"barbell","barn","bassoon","bath_towel","beach_wagon",
"bicycle-built-for-two","binoculars","boathouse","bonnet","bookcase",
"bow","brass","breastplate","bullet_train","cannon",
"can_opener","carpenter’s_kit","cassette","cellular_telephone","chain_saw",
"chest","china_cabinet","clog","combination_lock","container_ship",
"corkscrew","crate","Crock_Pot","digital_watch","dining_table",
"dishwasher","doormat","Dutch_oven","electric_fan","electric_locomotive",
"envelope","file","folding_chair","football_helmet","freight_car",
"French_horn","fur_coat","garbage_truck","goblet","golf_ball",
"grand_piano","half_track","hamper","hard_disc","harmonica",
"harvester","hook","horizontal_bar","horse_cart","iron","jack-o’-lantern",
"lab_coat","ladle","letter_opener","liner","mailbox","megalith",
"military_uniform","milk_can","mixing_bowl","monastery","mortar",
"mosquito_net","motor_scooter","mountain_bike","mountain_tent","mousetrap",
"necklace","nipple","ocarina","padlock","palace",
"parallel_bars","park_bench","pedestal","pencil_sharpener","pickelhaube",
"pillow","planetarium","plastic_bag","Polaroid_camera","pole",
"pot","purse","quilt","radiator","radio",
"radio_telescope","rain_barrel","reflex_camera","refrigerator","rifle",
"rocking_chair","rubber_eraser","rule","running_shoe","sewing_machine",
"shield","shoji","ski","ski_mask","slot",
"soap_dispenser","soccer_ball","sock","soup_bowl","space_heater",
"spider_web","spindle","sports_car","steel_arch_bridge","stethoscope",
"streetcar","submarine","swimming_trunks","syringe","table_lamp",
"tank","teddy","television","throne","tile_roof",
"toilet_seat","trench_coat","trimaran","typewriter_keyboard","umbrella",
"vase","volleyball","wardrobe","warplane","washer",
"water_bottle","water_tower","whiskey_jug","wig","wine_bottle",
"wok","wreck","yawl","yurt","street_sign","traffic_light",
"consomme","ice_cream","bagel","cheeseburger","hotdog",
"mashed_potato","spaghetti_squash","bell_pepper","cardoon",
 "Granny_Smith","strawberry","lemon","carbonara","burrito",
"cup","coral_reef","yellow_lady’s_slipper","buckeye","agaric","gyromitra","earthstar","bolete"]


all_texts_919=["tench","goldfish","great_white_shark","tiger_shark","hammerhead",
"electric_ray","stingray","cock","hen","ostrich",
"brambling","goldfinch","house_finch","junco","indigo_bunting",
"robin","bulbul","jay","magpie","chickadee",
"water_ouzel","kite","bald_eagle","vulture","great_grey_owl",
"European_fire_salamander","common_newt","eft","spotted_salamander","axolotl",
"bullfrog","tree_frog","tailed_frog","loggerhead","leatherback_turtle",
"mud_turtle","terrapin","box_turtle","banded_gecko","common_iguana",
"American_chameleon","whiptail","agama","frilled_lizard","alligator_lizard",
"Gila_monster","green_lizard","African_chameleon","Komodo_dragon","African_crocodile",
"American_alligator","triceratops","thunder_snake","ringneck_snake","hognose_snake",
"green_snake","king_snake","garter_snake","water_snake","vine_snake",
"night_snake","boa_constrictor","rock_python","Indian_cobra","green_mamba",
"sea_snake","horned_viper","diamondback","sidewinder","trilobite",
"harvestman","scorpion","black_and_gold_garden_spider","barn_spider","garden_spider",
"black_widow","tarantula","wolf_spider","tick","centipede",
"black_grouse","ptarmigan","ruffed_grouse","prairie_chicken","peacock",
"quail","partridge","African_grey","macaw","sulphur-crested_cockatoo",
"lorikeet","coucal","bee_eater","hornbill","hummingbird",
"jacamar","toucan","drake","red-breasted_merganser","goose",
"black_swan","tusker","echidna","platypus","wallaby",
"koala","wombat","jellyfish","sea_anemone","brain_coral",
"flatworm","nematode","conch","snail","slug",
"sea_slug","chiton","chambered_nautilus","Dungeness_crab","rock_crab",
"fiddler_crab","king_crab","American_lobster","spiny_lobster","crayfish",
"hermit_crab","isopod","white_stork","black_stork","spoonbill",
"flamingo","little_blue_heron","American_egret","bittern","crane",
"limpkin","European_gallinule","American_coot","bustard","ruddy_turnstone",
"red-backed_sandpiper","redshank","dowitcher","oystercatcher","pelican",
"king_penguin","albatross","grey_whale","killer_whale","dugong",
"sea_lion","Chihuahua","Japanese_spaniel","Maltese_dog","Pekinese",
"Shih-Tzu","Blenheim_spaniel","papillon","toy_terrier","Rhodesian_ridgeback",
"Afghan_hound","basset","beagle","bloodhound","bluetick",
"black-and-tan_coonhound","Walker_hound","English_foxhound","redbone","borzoi",
"Irish_wolfhound","Italian_greyhound","whippet","Ibizan_hound","Norwegian_elkhound",
"otterhound","Saluki","Scottish_deerhound","Weimaraner","Staffordshire_bullterrier",
"American_Staffordshire_terrier","Bedlington_terrier","Border_terrier","Kerry_blue_terrier","Irish_terrier",
"Norfolk_terrier","Norwich_terrier","Yorkshire_terrier","wire-haired_fox_terrier","Lakeland_terrier",
"Sealyham_terrier","Airedale","cairn","Australian_terrier","Dandie_Dinmont",
"Boston_bull","miniature_schnauzer","giant_schnauzer","standard_schnauzer","Scotch_terrier",
"Tibetan_terrier","silky_terrier","soft-coated_wheaten_terrier","West_Highland_white_terrier","Lhasa",
"flat-coated_retriever","curly-coated_retriever","golden_retriever","Labrador_retriever","Chesapeake_Bay_retriever",
"German_short-haired_pointer","vizsla","English_setter","Irish_setter","Gordon_setter",
"Brittany_spaniel","clumber","English_springer","Welsh_springer_spaniel","cocker_spaniel",
"Sussex_spaniel","Irish_water_spaniel","kuvasz","schipperke","groenendael",
"malinois","briard","kelpie","komondor","Old_English_sheepdog",
"Shetland_sheepdog","collie","Border_collie","Bouvier_des_Flandres","Rottweiler",
"German_shepherd","Doberman","miniature_pinscher","Greater_Swiss_Mountain_dog","Bernese_mountain_dog",
"Appenzeller","EntleBucher","boxer","bull_mastiff","Tibetan_mastiff",
"French_bulldog","Great_Dane","Saint_Bernard","Eskimo_dog","malamute",
"Siberian_husky","dalmatian","affenpinscher","basenji","pug",
"Leonberg","Newfoundland","Great_Pyrenees","Samoyed","Pomeranian",
"chow","keeshond","Brabancon_griffon","Pembroke","Cardigan",
"toy_poodle","miniature_poodle","standard_poodle","Mexican_hairless","timber_wolf",
"white_wolf","red_wolf","coyote","dingo","dhole",
"African_hunting_dog","hyena","red_fox","kit_fox","Arctic_fox",
"grey_fox","tabby","tiger_cat","Persian_cat","Siamese_cat",
"Egyptian_cat","cougar","lynx","leopard","snow_leopard",
"jaguar","lion","tiger","cheetah","brown_bear",
"American_black_bear","ice_bear","sloth_bear","mongoose","meerkat",
"tiger_beetle","ladybug","ground_beetle","long-horned_beetle","leaf_beetle",
"dung_beetle","rhinoceros_beetle","weevil","fly","bee",
"ant","grasshopper","cricket","walking_stick","cockroach",
"mantis","cicada","leafhopper","lacewing","dragonfly",
"damselfly","admiral","ringlet","monarch","cabbage_butterfly",
"sulphur_butterfly","lycaenid","starfish","sea_urchin","sea_cucumber",
"wood_rabbit","hare","Angora","hamster","porcupine",
"fox_squirrel","marmot","beaver","guinea_pig","sorrel",
"zebra","hog","wild_boar","warthog","hippopotamus",
"ox","water_buffalo","bison","ram","bighorn",
"ibex","hartebeest","impala","gazelle","Arabian_camel",
"llama","weasel","mink","polecat","black-footed_ferret",
"otter","skunk","badger","armadillo","three-toed_sloth",
"orangutan","gorilla","chimpanzee","gibbon","siamang",
"guenon","patas","baboon","macaque","langur",
"colobus","proboscis_monkey","marmoset","capuchin","howler_monkey",
"titi","spider_monkey","squirrel_monkey","Madagascar_cat","indri",
"Indian_elephant","African_elephant","lesser_panda","giant_panda","barracouta",
"eel","coho","rock_beauty","anemone_fish","sturgeon",
"gar","lionfish","puffer","abacus","abaya",
"academic_gown","accordion","acoustic_guitar","aircraft_carrier","airliner",
"airship","ambulance","amphibian","analog_clock","apiary",
"apron","ashcan","assault_rifle","backpack","balloon",
"ballpoint","Band_Aid","banjo","barbell","barber_chair",
"barn","barometer","barrel","barrow","baseball",
"basketball","bassinet","bassoon","bath_towel","bathtub",
"beach_wagon","beacon","beaker","bearskin","beer_bottle",
"beer_glass","bib","bicycle-built-for-two","binder","binoculars",
"birdhouse","boathouse","bobsled","bolo_tie","bonnet",
"bookcase","bow","bow_tie","brass","brassiere",
"breastplate","broom","bucket","buckle","bulletproof_vest",
"bullet_train","cab","caldron","candle","cannon",
"canoe","can_opener","cardigan","car_mirror","carpenter’s_kit",
"carton","cassette","cassette_player","castle","catamaran",
"cello","cellular_telephone","chain","chainlink_fence","chain_saw",
"chest","chiffonier","chime","china_cabinet","Christmas_stocking",
"church","cleaver","cloak","clog","cocktail_shaker",
"coffee_mug","coffeepot","combination_lock","container_ship","convertible",
"corkscrew","cornet","cowboy_boot","cowboy_hat","cradle",
"crane","crash_helmet","crate","crib","Crock_Pot",
"croquet_ball","crutch","cuirass","desk","dial_telephone",
"diaper","digital_clock","digital_watch","dining_table","dishrag",
"dishwasher","doormat","drilling_platform","drum","drumstick",
"dumbbell","Dutch_oven","electric_fan","electric_guitar","electric_locomotive",
"envelope","espresso_maker","face_powder","feather_boa","file",
"fireboat","fire_engine","fire_screen","flagpole","flute",
"folding_chair","football_helmet","forklift","fountain_pen","four-poster",
"freight_car","French_horn","frying_pan","fur_coat","garbage_truck",
"gasmask","gas_pump","goblet","go-kart","golf_ball",
"golfcart","gondola","gong","gown","grand_piano",
"guillotine","hair_slide","hair_spray","half_track","hammer",
"hamper","hand_blower","hand-held_computer","handkerchief","hard_disc",
"harmonica","harp","harvester","hatchet","holster",
"honeycomb","hook","hoopskirt","horizontal_bar","horse_cart",
"hourglass","iPod","iron","jack-o’-lantern","jean",
"jeep","jersey","jigsaw_puzzle","jinrikisha","joystick",
"kimono","knee_pad","knot","lab_coat","ladle",
"lawn_mower","lens_cap","letter_opener","lifeboat","lighter",
"limousine","liner","lipstick","Loafer","lotion",
"loudspeaker","loupe","magnetic_compass","mailbag","mailbox",
"manhole_cover","maraca","marimba","mask","matchstick",
"maypole","measuring_cup","medicine_chest","megalith","microphone",
"microwave","military_uniform","milk_can","minibus","miniskirt",
"minivan","missile","mitten","mixing_bowl","mobile_home",
"Model_T","modem","monastery","monitor","moped",
"mortar","mortarboard","mosque","mosquito_net","motor_scooter",
"mountain_bike","mountain_tent","mouse","mousetrap","moving_van",
"muzzle","nail","neck_brace","necklace","nipple",
"notebook","obelisk","oboe","ocarina","odometer",
"oil_filter","organ","oscilloscope","overskirt","oxcart",
"oxygen_mask","paddle","paddlewheel","padlock","paintbrush",
"pajama","palace","panpipe","parachute","parallel_bars",
"park_bench","parking_meter","passenger_car","pay-phone","pedestal",
"pencil_box","pencil_sharpener","perfume","Petri_dish","photocopier",
"pick","pickelhaube","picket_fence","pickup","pier",
"piggy_bank","pill_bottle","pillow","ping-pong_ball","pinwheel",
"pirate","pitcher","plane","planetarium","plastic_bag",
"plate_rack","plunger","Polaroid_camera","pole","police_van",
"poncho","pool_table","pop_bottle","pot","potter’s_wheel",
"power_drill","prayer_rug","printer","prison","projector",
"puck","punching_bag","purse","quill","quilt",
"racer","racket","radiator","radio","radio_telescope",
"rain_barrel","recreational_vehicle","reel","reflex_camera","refrigerator",
"remote_control","revolver","rifle","rocking_chair","rubber_eraser",
"rugby_ball","rule","running_shoe","safe","safety_pin",
"saltshaker","sandal","sarong","sax","scabbard",
"scale","school_bus","schooner","scoreboard","screw",
"screwdriver","seat_belt","sewing_machine","shield","shoji",
"shopping_basket","shopping_cart","shovel","shower_cap","shower_curtain",
"ski","ski_mask","sleeping_bag","slide_rule","slot",
"snowmobile","snowplow","soap_dispenser","soccer_ball","sock",
"solar_dish","sombrero","soup_bowl","space_heater","space_shuttle",
"spatula","speedboat","spider_web","spindle","sports_car",
"spotlight","steam_locomotive","steel_arch_bridge","steel_drum","stethoscope",
"stole","stone_wall","stopwatch","stove","strainer",
"streetcar","stretcher","studio_couch","stupa","submarine",
"suit","sundial","sunglass","suspension_bridge","swab",
"sweatshirt","swimming_trunks","swing","syringe","table_lamp",
"tank","teapot","teddy","television","tennis_ball",
"thatch","theater_curtain","thimble","thresher","throne",
"tile_roof","toaster","toilet_seat","torch","totem_pole",
"tow_truck","tractor","trailer_truck","tray","trench_coat",
"tricycle","trimaran","tripod","triumphal_arch","trolleybus",
"trombone","typewriter_keyboard","umbrella","unicycle","upright",
"vacuum","vase","velvet","vending_machine","vestment",
"viaduct","violin","volleyball","waffle_iron","wall_clock",
"wallet","wardrobe","warplane","washbasin","washer",
"water_bottle","water_jug","water_tower","whiskey_jug","whistle",
"wig","window_screen","window_shade","Windsor_tie","wine_bottle",
"wok","wooden_spoon","worm_fence","wreck","yawl",
"yurt","comic_book","street_sign","traffic_light","menu",
"plate","guacamole","consomme","trifle","ice_cream",
"ice_lolly","French_loaf","bagel","pretzel","cheeseburger",
"hotdog","mashed_potato","head_cabbage","broccoli","cauliflower",
"zucchini","spaghetti_squash","acorn_squash","butternut_squash","cucumber",
"artichoke","bell_pepper","cardoon","mushroom","Granny_Smith",
"strawberry","orange","lemon","fig","pineapple",
"banana","jackfruit","custard_apple","pomegranate","hay",
"carbonara","dough","meat_loaf","pizza","potpie",
"burrito","cup","eggnog","bubble","cliff",
"coral_reef","ballplayer","scuba_diver","rapeseed","daisy",
"yellow_lady’s_slipper","corn","acorn","hip","buckeye",
"coral_fungus","agaric","gyromitra","stinkhorn","earthstar",
"hen-of-the-woods","bolete","ear","toilet_tissue"
]

class ImageNetSDataset(Dataset):
    """
    ImageNet-S Dataset for semantic segmentation

    Dataset structure:
    ├── ImageNetS50/300/919
        ├── train
        ├── train-semi (10 images per class with pixel annotations)
        ├── train-semi-segmentation
        ├── validation
        ├── validation-segmentation
        └── test

    Annotations are stored as PNG with RGB channels:
    - Class ID = R + G*256
    - Ignored region = 1000
    - Other category = 0
    """

    def __init__(self,
                 root_dir,
                 split='train',
                 variant='ImageNetS50',
                 use_semi=False,
                 transform=None,
                 return_mask=False,
                 samples_per_class=None):
        """
        Args:
            root_dir: Root directory containing ImageNetS datasets
            split: 'train', 'validation', or 'test'
            variant: 'ImageNetS50', 'ImageNetS300', or 'ImageNetS919'
            use_semi: Use semi-supervised annotations (train-semi)
            transform: Image transformations
            return_mask: Return segmentation mask if available
        """
        self.root_dir = os.path.join(root_dir, variant)
        self.split = split
        self.variant = variant
        self.use_semi = use_semi
        self.transform = transform
        self.return_mask = return_mask
        self.samples_per_class = samples_per_class

        # Determine image and mask directories
        if split == 'train' and use_semi:
            self.image_dir = os.path.join(self.root_dir, 'train-semi')
            self.mask_dir = os.path.join(self.root_dir, 'train-semi-segmentation')
        elif split == 'validation':
            self.image_dir = os.path.join(self.root_dir, 'validation')
            self.mask_dir = os.path.join(self.root_dir, 'validation-segmentation')
        else:
            self.image_dir = os.path.join(self.root_dir, split)
            self.mask_dir = None

        # Get class names and number
        self.num_classes = self._get_num_classes()
        self.class_names = self._load_class_names()

        # Load image paths
        if samples_per_class is not None and split == 'train':
            self.samples = self._load_samples(self.samples_per_class)
        else:
            self.samples = self._load_samples(None)
        print(f"Loaded {len(self.samples)} samples from {variant} {split}")

    def _get_num_classes(self):
        """Get number of classes based on variant"""
        num_classes_map = {
            'ImageNetS50': 50,
            'ImageNetS300': 300,
            'ImageNetS919': 919
        }
        return num_classes_map.get(self.variant, 50)

    def _load_class_names(self):
        """Load class names from directory structure"""
        class_dirs = sorted([d for d in os.listdir(self.image_dir)
                             if os.path.isdir(os.path.join(self.image_dir, d))])
        return class_dirs

    def _load_samples(self,samples_per_class):
        """Load all image paths and corresponding info"""
        import random
        samples = []

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.image_dir, class_name)

            if not os.path.isdir(class_dir):
                continue

            # Get all images in class directory
            if samples_per_class is not None:
                try:
                    images0 = [f for f in os.listdir(class_dir)
                               if f.lower().endswith(('.jpeg', '.jpg', '.png', '.JPEG'))]
                    images = random.sample(images0, samples_per_class)
                except FileNotFoundError:
                    print(f"Warning: samples_per_class directory not found: {class_dir}")
                    continue
            else:
                try:
                    images = [f for f in os.listdir(class_dir)
                              if f.lower().endswith(('.jpeg', '.jpg', '.png', '.JPEG'))]
                except FileNotFoundError:
                    print(f"Warning: Class directory not found: {class_dir}")
                    continue



            for img_name in images:
                img_path = os.path.join(class_dir, img_name)

                # Check if image exists
                if not os.path.exists(img_path):
                    continue

                # Determine mask path if available
                mask_path = None
                has_mask = False
                if self.mask_dir is not None:
                    mask_name = os.path.splitext(img_name)[0] + '.png'
                    mask_path = os.path.join(self.mask_dir, class_name, mask_name)
                    if os.path.exists(mask_path):
                        has_mask = True

                # Get class name based on variant
                if self.variant == 'ImageNetS50':
                    class_name_text = all_texts_50[class_idx] if class_idx < len(all_texts_50) else class_name
                elif self.variant == 'ImageNetS300':
                    class_name_text = all_texts_300[class_idx] if class_idx < len(all_texts_300) else class_name
                elif self.variant == 'ImageNetS919':
                    class_name_text = all_texts_919[class_idx] if class_idx < len(all_texts_919) else class_name
                else:
                    class_name_text = class_name

                samples.append({
                    'image_path': img_path,
                    'mask_path': mask_path if has_mask else None,
                    'class_idx': class_idx,
                    'class_name': class_name_text,
                    'image_name': img_name,
                })

        return samples

    def _decode_mask(self, mask_img):
        """
        Decode ImageNet-S mask from PNG

        Class ID = R + G*256
        Ignored = 1000
        Other = 0

        Args:
            mask_img: PIL Image with RGB channels

        Returns:
            mask: Numpy array with class indices [H, W]
        """
        mask_array = np.array(mask_img)

        # Extract R and G channels
        R = mask_array[:, :, 0].astype(np.int32)
        G = mask_array[:, :, 1].astype(np.int32)

        # Decode class ID
        mask = R + G * 256

        # Handle special values
        mask[mask == 1000] = -1  # Ignore region
        mask[mask == 0] = -1  # Other category (treated as ignore)

        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        # Load image
        try:
            # Load image
            # image = Image.open(sample_info['image_path']).convert('RGB')
            # modified by jojo to solve input image mistakes
            image = Image.open(sample_info['image_path'])
            # print("shape:", np.array(image).shape)
            if len(np.array(image).shape) == 3 and np.array(image).shape[2] > 3:
                image = image.convert("RGB")
                # 此处出现问题，有些图像具有四个通道，如img_path: train1/n02747177/n02747177_10752.JPEG，就会在图像预处理环节出现维度不匹配的error
            # add-----3channel misatake 20231114
            if len(np.array(image).shape) == 2:
                image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
                image = Image.fromarray(image.astype("uint8"))
        except Exception as e:
            print(f"Error loading image {sample_info['image_path']}: {e}")
            # Return a dummy image if loading fails

        original_size = image.size  # return tuple (width, height)

        # Apply transforms to image
        if self.transform is not None:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error transforming image {sample_info['image_path']}: {e}")
                # Create a dummy tensor as fallback
                image = torch.zeros(3, 512, 512)

        # Prepare return dict
        item = {
            'image': image,
            'class_idx': torch.tensor(sample_info['class_idx']).long(),
            'class_name': sample_info['class_name'],
            'image_path': sample_info['image_path'],
            'original_size': torch.tensor(original_size),
        }

        # Only return mask if requested and available
        if self.return_mask and sample_info.get('mask_path') is not None:
            try:
                mask_img = Image.open(sample_info['mask_path'])
                mask = self._decode_mask(mask_img)

                # Resize mask if transform was applied
                if self.transform is not None and isinstance(image, torch.Tensor):
                    # Resize mask to match transformed image
                    h, w = image.shape[1], image.shape[2]
                    mask_img_resized = Image.fromarray(mask.astype(np.uint8))
                    mask_img_resized = mask_img_resized.resize((w, h), Image.NEAREST)
                    mask = np.array(mask_img_resized)

                item['mask'] = torch.from_numpy(mask).long()
                item['has_mask'] = torch.tensor(True)
            except Exception as e:
                print(f"Error loading mask {sample_info.get('mask_path')}: {e}")
                item['mask'] = torch.full((image.shape[1] if isinstance(image, torch.Tensor) else 512,
                                           image.shape[2] if isinstance(image, torch.Tensor) else 512), -1,
                                          dtype=torch.long)
                item['has_mask'] = torch.tensor(False)
        elif self.return_mask:
            # Return a dummy mask with all -1
            item['mask'] = torch.full((image.shape[1] if isinstance(image, torch.Tensor) else 512,
                                       image.shape[2] if isinstance(image, torch.Tensor) else 512), -1,
                                      dtype=torch.long)
            item['has_mask'] = torch.tensor(False)
        # 如果不需要mask，则不返回mask相关字段
        # print("item:", item)

        return item


def get_imagenet_s_dataloaders(root_dir,
                               variant='ImageNetS50',
                               batch_size=8,
                               num_workers=4,
                               image_size=512):
    """
    Create train and validation dataloaders for ImageNet-S

    Args:
        root_dir: Root directory containing ImageNetS datasets
        variant: 'ImageNetS50', 'ImageNetS300', or 'ImageNetS919'
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size for resizing

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        class_names: List of class names
    """
    from torchvision.transforms import InterpolationMode

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    # Create datasets
    # Training: 无监督，不需要mask
    train_dataset = ImageNetSDataset(
        root_dir=root_dir,
        split='train',
        variant=variant,
        use_semi=False,  # Use full training set without annotations
        transform=train_transform,
        return_mask=False  # 训练时不需要mask
    )

    # Validation: 有监督，需要mask
    val_dataset = ImageNetSDataset(
        root_dir=root_dir,
        split='validation',
        variant=variant,
        transform=val_transform,
        return_mask=True  # 验证时需要mask
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.class_names


def get_semi_supervised_loader(root_dir,
                               variant='ImageNetS50',
                               batch_size=8,
                               num_workers=4,
                               image_size=512):
    """
    Create dataloader for semi-supervised training (10 images per class)

    Returns:
        semi_loader: Semi-supervised dataloader with masks
    """
    from torchvision.transforms import InterpolationMode

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    semi_dataset = ImageNetSDataset(
        root_dir=root_dir,
        split='train',
        variant=variant,
        use_semi=True,
        transform=transform,
        return_mask=True
    )

    semi_loader = DataLoader(
        semi_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return semi_loader