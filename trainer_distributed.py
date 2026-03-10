"""
Multi-GPU Distributed Trainer for PLUSS_β
Supports DDP, gradient accumulation, and mixed precision training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import sys
from collections import defaultdict
import cv2
import numpy as np

# Import models
from pluss_beta.models.semantic_tuner import SemanticTuner, SemanticTunerLoss
from pluss_beta.models.box_tuner import BoxTuner, BoxTunerLoss
from pluss_beta.models.memory_bank import MemoryBank
from pluss_beta.utils.point2box import Point2BoxConverter
from pluss_beta.utils.distributed import (
    is_main_process, get_rank, get_world_size, barrier,
    reduce_dict, GradientAccumulator, AverageMeter,
    save_checkpoint_distributed, print_on_main
)

# CLIP Surgery for feature extraction
sys.path.insert(0, 'CLIP_Surgery')
import CLIP_Surgery.clip as clips
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator,sam_model_registry


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


class DistributedPLUSSBetaTrainer:
    """
    Multi-GPU Distributed Trainer for PLUSS_β

    Key Features:
    - DistributedDataParallel (DDP) support
    - Gradient accumulation for large effective batch sizes
    - Mixed precision training (FP16)
    - Synchronized BatchNorm
    - CRITICAL: Maintains separate computational graphs for tuners
    """

    def __init__(self,
                 clip_model,
                 sam_model,
                 sam_checkpoint,
                 grounding_dino,
                 config,
                 local_rank=0,
                 world_size=1):
        """
        Args:
            clip_model: Frozen CLIP model
            sam_model: Frozen SAM model
            grounding_dino: Grounding DINO model
            config: Training configuration
            local_rank: Local GPU rank
            world_size: Total number of GPUs
        """
        self.local_rank = local_rank
        self.world_size = world_size
        self.config = config
        self.device = torch.device(f'cuda:{local_rank}')

        # Frozen foundation models
        self.clip_model = clip_model.to(self.device).eval()
        self.sam_model = sam_model.to(self.device).eval()
        # Use sam_model directly for batch processing (not sam_predictor)
        from segment_anything import build_sam
        self.sam_model = build_sam(checkpoint=sam_checkpoint).to(self.device).eval()
        # self.sam_predictor=SamPredictor(build_sam(checkpoint=sam_checkpoint).to(self.device))
        self.grounding_dino = grounding_dino.to(self.device).eval()

        # Freeze foundation models
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.sam_model.parameters():
            param.requires_grad = False
        for param in self.grounding_dino.parameters():
            param.requires_grad = False

        # Initialize trainable components
        self.semantic_tuner = SemanticTuner(
            num_layers=config.get('num_layers', 12),
            embed_dim=config.get('embed_dim', 512),
            num_prompts=config.get('num_prompts', 16),
            dropout=config.get('dropout', 0.1)
        ).to(self.device)

        self.box_tuner = BoxTuner(
            feature_dim=config.get('feature_dim', 512),
            num_heads=config.get('num_heads', 8),
            hidden_dim=config.get('hidden_dim', 2048),
            dropout=config.get('dropout', 0.1)
        ).to(self.device)

        # Wrap with DDP (CRITICAL: Separate DDP wrappers for isolation)
        if world_size > 1:
            self.semantic_tuner = DDP(
                self.semantic_tuner,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False  # Strict mode for gradient checking
            )

            self.box_tuner = DDP(
                self.box_tuner,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )

        # Memory bank (shared across GPUs, synchronized)
        self.memory_bank = MemoryBank(
            capacity=config.get('memory_capacity', 1000),
            threshold=config.get('hard_threshold', 0.5),
            alpha=config.get('alpha', 0.7),
            beta=config.get('beta', 0.3)
        )

        # Point-to-box converter
        self.point2box = Point2BoxConverter(
            min_pts=config.get('min_pts', 3),
            image_size=(config.get('image_size', 512), config.get('image_size', 512))
        )

        # Loss functions
        self.semantic_loss_fn = SemanticTunerLoss(
            temperature=config.get('temperature', 0.07)
        ).to(self.device)

        # self.box_loss_fn = BoxTunerLoss(
        #     lambda_l1=config.get('lambda_l1', 1.0),
        #     lambda_giou=config.get('lambda_giou', 2.0)
        # ).to(self.device)
        #修改Box Loss初始化，添加image_size参数
        self.box_loss_fn = BoxTunerLoss(
            lambda_l1=config.get('lambda_l1', 0.5),  # 从1.0降到0.5
            lambda_giou=config.get('lambda_giou', 1.0),  # 从2.0降到1.0
            image_size=config.get('image_size', 512)  # 添加image_size
        ).to(self.device)

        # CRITICAL: Separate optimizers for separate computational graphs
        self.semantic_optimizer = optim.AdamW(
            self.semantic_tuner.parameters(),
            lr=config.get('semantic_lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )

        self.box_optimizer = optim.AdamW(
            self.box_tuner.parameters(),
            lr=config.get('box_lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )

        # Learning rate schedulers
        self.semantic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.semantic_optimizer,
            T_max=config.get('num_epochs', 1000),
            eta_min=1e-6
        )

        self.box_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.box_optimizer,
            T_max=config.get('num_epochs', 1000),
            eta_min=1e-6
        )

        # Gradient accumulation
        self.accumulation_steps = config.get('accumulation_steps', 1)
        self.semantic_accumulator = GradientAccumulator(
            self.semantic_tuner, self.accumulation_steps
        )
        self.box_accumulator = GradientAccumulator(
            self.box_tuner, self.accumulation_steps
        )

        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.semantic_scaler = GradScaler(enabled=self.use_amp)
        self.box_scaler = GradScaler(enabled=self.use_amp)

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Metrics tracking
        self.train_metrics = defaultdict(AverageMeter)

        # 最后添加：初始化text features缓存
        self._init_text_features_cache()

        # 预计算denormalization参数
        self._denorm_mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self._denorm_std = np.array([0.26862954, 0.26130258, 0.27577711])

    # ==========================================
    # 添加缓存初始化方法
    # ==========================================

    def _init_text_features_cache(self):
        """
        核心优化：预计算text features，避免每个batch重复编码

        节省时间: ~2-3小时/epoch
        """
        if is_main_process():
            print("\n" + "=" * 60)
            print("Initializing Text Features Cache...")
            print("=" * 60)

        self.cached_text_features = {}
        self.cached_redundant_features = None

        # 定义所有variants的文本列表
        text_lists = {
            'ImageNetS50': all_texts_50,
            'ImageNetS300': all_texts_300,
            'ImageNetS919': all_texts_919
        }

        with torch.no_grad():
            # 1. 预计算redundant features (只需一次)
            if is_main_process():
                print("  [1/2] Computing redundant features...")
            self.cached_redundant_features = clips.encode_text_with_prompt_ensemble(
                self.clip_model, [""], self.device
            )

            # 2. 预计算每个variant的text features
            if is_main_process():
                print("  [2/2] Computing text features for each variant...")

            for variant, text_list in text_lists.items():
                f_T = clips.encode_text_with_prompt_ensemble(
                    self.clip_model, text_list, self.device
                )
                self.cached_text_features[variant] = f_T

                if is_main_process():
                    memory_mb = f_T.numel() * f_T.element_size() / 1024 / 1024
                    print(f"    ✅ {variant}: {f_T.shape} ({memory_mb:.2f} MB)")

        # 设置当前variant
        self.current_variant = 'ImageNetS50'  # 默认

        if is_main_process():
            total_memory = sum(v.numel() * v.element_size() for v in self.cached_text_features.values())
            total_memory += self.cached_redundant_features.numel() * self.cached_redundant_features.element_size()
            total_memory_mb = total_memory / 1024 / 1024
            print(f"\n  Total cache size: {total_memory_mb:.2f} MB")
            print("=" * 60 + "\n")

    def set_variant(self, variant):
        """切换variant（如果需要训练不同规模的数据集）
        """
        if variant in self.cached_text_features:
            self.current_variant = variant
            if is_main_process():
                print(f"Switched to variant: {variant}")
        else:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(self.cached_text_features.keys())}")

    def train_box_tuner_step(self, images, texts, B_init, B_pseudo, sam_tokens, f_I):
        """
        Train Box Tuner with gradient accumulation and mixed precision

        CRITICAL: Completely separate from semantic tuner gradient path

        Args:
            images: Input images [B, 3, H, W]
            texts: Target texts (not used directly)
            B_init: List of initial boxes for each image
            B_pseudo: Concatenated pseudo boxes [total_boxes, 4]
            sam_tokens: List of SAM tokens for each image
            f_I: CLIP image features [B, 1025, 512]
        """
        # Extract CLIP region features (detached from semantic path)
        with autocast(enabled=self.use_amp):
            clip_feature_map = self.get_clip_feature_map(images, f_I.detach())

            # Extract ROI features - now handles List of boxes
            clip_region_features = (self.box_tuner.module if self.world_size > 1
                                    else self.box_tuner).extract_roi_features(
                clip_feature_map, B_init
            )

            # Concatenate sam_tokens from list
            if isinstance(sam_tokens, list):
                sam_tokens_list = []
                for tokens in sam_tokens:
                    if len(tokens) > 0:
                        sam_tokens_list.append(tokens)
                if len(sam_tokens_list) > 0:
                    sam_tokens_concat = torch.cat(sam_tokens_list, dim=0)  # [total_boxes, sam_dim]
                else:
                    # No tokens, skip this step
                    return 0.0, {}
            else:
                sam_tokens_concat = sam_tokens

            # Concatenate B_init for later use
            if isinstance(B_init, list):
                B_init_list = []
                for boxes in B_init:
                    if len(boxes) > 0:
                        # Ensure boxes are on correct device
                        if boxes.device != self.device:
                            boxes = boxes.to(self.device)
                        B_init_list.append(boxes)
                if len(B_init_list) > 0:
                    B_init_concat = torch.cat(B_init_list, dim=0)
                else:
                    return 0.0, {}
            else:
                # Ensure single tensor is on correct device
                if B_init.device != self.device:
                    B_init = B_init.to(self.device)
                B_init_concat = B_init

            # Ensure B_pseudo on correct device
            if B_pseudo.device != self.device:
                B_pseudo = B_pseudo.to(self.device)

            # Ensure same number of features across ALL sources
            num_features = min(
                clip_region_features.shape[0],
                sam_tokens_concat.shape[0],
                B_init_concat.shape[0],
                B_pseudo.shape[0]  # ← Add B_pseudo to the min calculation
            )

            if num_features == 0:
                return 0.0, {}

            # Match all tensors to same size
            clip_region_features = clip_region_features[:num_features]
            sam_tokens_concat = sam_tokens_concat[:num_features]
            B_init_concat = B_init_concat[:num_features]
            B_pseudo_matched = B_pseudo[:num_features]

            # Forward through box tuner
            fused_features = (self.box_tuner.module if self.world_size > 1
                              else self.box_tuner)(sam_tokens_concat.detach(), clip_region_features)

            # Predict box adjustments
            delta_boxes = (self.box_tuner.module if self.world_size > 1
                           else self.box_tuner).predict_box_adjustment(fused_features)

            # Refine boxes
            B_ref = (self.box_tuner.module if self.world_size > 1
                     else self.box_tuner).refine_boxes(B_init_concat, delta_boxes)

            # Compute box loss
            loss, metrics = self.box_loss_fn(B_ref, B_pseudo_matched)

            # Scale loss for gradient accumulation
            loss = self.box_accumulator.scale_loss(loss)

        # Backward with gradient scaling
        self.box_scaler.scale(loss).backward()

        # Update only when accumulation is complete
        if self.box_accumulator.should_update():
            # Gradient clipping
            self.box_scaler.unscale_(self.box_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.box_tuner.parameters(), max_norm=1.0
            )

            # Optimizer step
            self.box_scaler.step(self.box_optimizer)
            self.box_scaler.update()
            self.box_optimizer.zero_grad()

        self.box_accumulator.step()

        return loss.item() * self.accumulation_steps, metrics

    def train_semantic_tuner_step(self, hard_examples_batch):
        """
        Train Semantic Tuner with mixed precision

        CRITICAL: Completely separate from box tuner gradient path
        """
        if hard_examples_batch is None:
            return 0.0

        # Move to device
        f_I = hard_examples_batch['f_I'].to(self.device)
        f_T = hard_examples_batch['f_T'].to(self.device)
        M_pseudo = hard_examples_batch['M_pseudo'].to(self.device)

        with autocast(enabled=self.use_amp):
            # Get adapted features through semantic tuner
            f_ST = (self.semantic_tuner.module if self.world_size > 1
                   else self.semantic_tuner).get_adapted_features(
                self.clip_model, f_I
            )

            # Compute alignment loss
            loss = self.semantic_loss_fn(f_ST, f_T, pred_mask=None, pseudo_mask=M_pseudo)

            # Scale loss for gradient accumulation
            loss = self.semantic_accumulator.scale_loss(loss)

        # Backward with gradient scaling
        self.semantic_scaler.scale(loss).backward()

        # Update only when accumulation is complete
        if self.semantic_accumulator.should_update():
            # Gradient clipping
            self.semantic_scaler.unscale_(self.semantic_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.semantic_tuner.parameters(), max_norm=1.0
            )

            # Optimizer step
            self.semantic_scaler.step(self.semantic_optimizer)
            self.semantic_scaler.update()
            self.semantic_optimizer.zero_grad()

        self.semantic_accumulator.step()

        return loss.item() * self.accumulation_steps

    def training_step(self, batch):
        """Single training step with distributed synchronization"""
        images = batch['image'].to(self.device)
        text_prompts = all_texts_50  # List of text prompt lists

        metrics = {}

        # Extract features and find target texts from CLIP
        with torch.no_grad():
            f_I, f_T, similarity, target_texts = self.extract_features(images, text_prompts)

        # Forward pass through two branches
        with torch.no_grad():
            # Branch 1: Point-prompt using CLIP similarity maps
            z, points, labels = self.forward_point_branch(
                images, text_prompts, f_I, similarity, target_texts
            )
            # Branch 2: Box-prompt using target texts from CLIP
            z_hat, B_init, sam_tokens = self.forward_box_branch(images, target_texts)

        # Hard example mining
        L_mask = self.memory_bank.compute_mask_loss(z, z_hat)

        # Add to memory bank if hard example
        if L_mask.mean().item() >= self.config.get('hard_threshold', 0.5):
            # Extract target class text features for each image
            # f_T is [num_classes, 512], need to select target classes
            batch_size = f_I.shape[0]

            # Get indices of target classes
            with torch.no_grad():
                similarity_cls = (100.0 * f_I[:, 0, :] @ f_T.T).softmax(dim=-1)  # [B, num_classes]
                _, target_indices = similarity_cls.topk(1, dim=-1)  # [B, 1]

            # Extract target text features for each image
            f_T_targets = f_T[target_indices.squeeze(-1)]  # [B, 512]

            # Add to memory bank with per-image text features
            self.memory_bank.add_entry(f_I, f_T_targets, z, z_hat)

        metrics['mask_loss'] = L_mask.mean().item()

        # Point-2-box conversion
        B_pseudo = self.point2box(points, labels)  # List of boxes for each image

        # Check if we have valid boxes for training
        # B_pseudo is list of arrays, B_init is list of tensors
        has_valid_boxes = False
        for i in range(len(B_pseudo)):
            if len(B_pseudo[i]) > 0 and len(B_init[i]) > 0:
                has_valid_boxes = True
                break

        if has_valid_boxes:
            # Convert B_pseudo list to proper format for box tuner
            # Concatenate all boxes from batch with batch indices
            B_pseudo_batch = []
            batch_indices = []
            for i, boxes in enumerate(B_pseudo):
                if len(boxes) > 0:
                    B_pseudo_batch.append(torch.tensor(boxes, device=self.device, dtype=torch.float32))
                    batch_indices.extend([i] * len(boxes))

            if len(B_pseudo_batch) > 0:
                B_pseudo_tensor = torch.cat(B_pseudo_batch, dim=0)

                # Train Box Tuner
                box_loss, box_metrics = self.train_box_tuner_step(
                    images, target_texts, B_init,
                    B_pseudo_tensor,
                    sam_tokens, f_I
                )
                metrics['box_loss'] = box_loss
                metrics.update({f'box_{k}': v for k, v in box_metrics.items()})

        # Train Semantic Tuner every 100 epochs
        if self.current_epoch % 50 == 0 and len(self.memory_bank) > 0:
        # if self.current_epoch % 100 == 0 and len(self.memory_bank) > 0:
            hard_batch = self.memory_bank.sample_batch(
                batch_size=self.config.get('semantic_batch_size', 32),
                prioritize_high_loss=True
            )
            semantic_loss = self.train_semantic_tuner_step(hard_batch)
            metrics['semantic_loss'] = semantic_loss

        self.global_step += 1

        return metrics

    def train_epoch(self, dataloader):
        """Train for one epoch with distributed data loading"""
        self.semantic_tuner.train()
        self.box_tuner.train()

        # Reset accumulators
        self.semantic_accumulator.reset()
        self.box_accumulator.reset()

        # Reset metrics
        for meter in self.train_metrics.values():
            meter.reset()

        # Progress bar only on main process
        if is_main_process():
            pbar = tqdm(dataloader, desc=f'Epoch {self.current_epoch}')
        else:
            pbar = dataloader

        for batch_idx, batch in enumerate(pbar):
            metrics = self.training_step(batch)

            # Update meters
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.train_metrics[k].update(v)

            # Update progress bar on main process
            if is_main_process() and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    k: f"{meter.avg:.4f}" for k, meter in self.train_metrics.items()
                })

        # Synchronize metrics across all GPUs
        for meter in self.train_metrics.values():
            meter.synchronize()

        # Get average metrics
        avg_metrics = {k: meter.avg for k, meter in self.train_metrics.items()}

        # Add memory bank stats (only on main process to avoid duplication)
        if is_main_process():
            mem_stats = self.memory_bank.get_statistics()
            avg_metrics.update({f'memory_{k}': v for k, v in mem_stats.items()})

        # Step schedulers
        self.semantic_scheduler.step()
        self.box_scheduler.step()

        self.current_epoch += 1

        return avg_metrics

    def save_checkpoint(self, path, is_best=False):
        """Save checkpoint (only on main process)"""
        if not is_main_process():
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Get model state dicts (unwrap DDP if necessary)
        semantic_state = (self.semantic_tuner.module.state_dict()
                         if self.world_size > 1
                         else self.semantic_tuner.state_dict())

        box_state = (self.box_tuner.module.state_dict()
                    if self.world_size > 1
                    else self.box_tuner.state_dict())

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'semantic_tuner': semantic_state,
            'box_tuner': box_state,
            'semantic_optimizer': self.semantic_optimizer.state_dict(),
            'box_optimizer': self.box_optimizer.state_dict(),
            'semantic_scheduler': self.semantic_scheduler.state_dict(),
            'box_scheduler': self.box_scheduler.state_dict(),
            'semantic_scaler': self.semantic_scaler.state_dict(),
            'box_scaler': self.box_scaler.state_dict(),
            'config': self.config
        }

        save_checkpoint_distributed(checkpoint, path, is_best)
        print_on_main(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load checkpoint on all processes"""
        # Map to correct device
        checkpoint = torch.load(path, map_location=self.device)

        # Load model states
        if self.world_size > 1:
            self.semantic_tuner.module.load_state_dict(checkpoint['semantic_tuner'])
            self.box_tuner.module.load_state_dict(checkpoint['box_tuner'])
        else:
            self.semantic_tuner.load_state_dict(checkpoint['semantic_tuner'])
            self.box_tuner.load_state_dict(checkpoint['box_tuner'])

        # Load optimizer states
        self.semantic_optimizer.load_state_dict(checkpoint['semantic_optimizer'])
        self.box_optimizer.load_state_dict(checkpoint['box_optimizer'])

        # Load schedulers
        if 'semantic_scheduler' in checkpoint:
            self.semantic_scheduler.load_state_dict(checkpoint['semantic_scheduler'])
        if 'box_scheduler' in checkpoint:
            self.box_scheduler.load_state_dict(checkpoint['box_scheduler'])

        # Load scalers
        if 'semantic_scaler' in checkpoint:
            self.semantic_scaler.load_state_dict(checkpoint['semantic_scaler'])
        if 'box_scaler' in checkpoint:
            self.box_scaler.load_state_dict(checkpoint['box_scaler'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        print_on_main(f"Checkpoint loaded from {path}")


    # Complete methods based on Text2Seg implementation
    def extract_features(self, images, text_prompts=None):
        """
        Extract features from CLIP (Equations 1, 2) and find target classes

        Args:
            images: Batch of images [B, 3, H, W]
            text_prompts: List of text prompt lists (batch_size x [class_names])

        Returns:
            f_I: Image features [B, 1025, 512]
            f_T: Text features [num_classes, 512]
            similarity: Similarity matrix [B, 1025, num_classes]
            target_texts: List of target class names for each image [B]
        """
        # with torch.no_grad():
        #     # Equation 1: f_I = ImageEncoder(I)
        #     f_I = self.clip_model.encode_image(images)
        #     f_I = f_I / f_I.norm(dim=1, keepdim=True)
        #     # print("f_I",f_I.shape)
        #     # Shape: [B, 1025, 512] where 1025 = 1(CLS) + 32*32(patches)   torch.Size([4, 1025, 512])
        #
        #     # Equation 2: f_T = TextEncoder(T)
        #     # Prompt ensemble for text features with normalization
        #     f_T = clips.encode_text_with_prompt_ensemble(self.clip_model, text_prompts, self.device)
        #     # print("f_T",f_T.shape)
        #     # Shape: [num_classes, 512]
        #
        #     # Extract redundant features from an empty string
        #     redundant_features = clips.encode_text_with_prompt_ensemble(self.clip_model, [""], self.device)
        #
        #     # CLIP feature surgery with custom redundant features
        #     similarity = clips.clip_feature_surgery(f_I, f_T, redundant_features)
        #     # print("similarity",similarity.shape)
        #     # Shape: [B, 1025, num_classes] [4,1025,50]
        #
        #     # Find target class for each image using CLS token similarity
        #     similarity1 = (100.0 * f_I[:, 0, :] @ f_T.T).softmax(dim=-1)  # [B, num_classes]
        #
        #     # Get top-1 class for each image
        #     values, indices = similarity1.topk(1, dim=-1)  # [B, 1] [4,1]
        #     #values, indices
        #     # tensor([[0.9950],
        #     #         [0.9834],
        #     #         [0.8182],
        #     #         [0.9984]], device='cuda:3')
        #     # tensor([[16],
        #     #         [16],
        #     #         [22],
        #     #         [8]], device='cuda:3')
        #     # 确保 indices 是整数类型
        #     indices = indices.squeeze(-1)  # 从 [B, 1] 变为 [B]
        #     # print("indices",indices)
        #
        #     # Extract target text for each image
        #     target_texts = []
        #     for i in range(indices.shape[0]):
        #         idx = indices[i].item()  # 现在是标量
        #         # 现在 text_prompts[idx] 应该是一个字符串，而不是列表
        #         target_texts.append(text_prompts[idx])

        # print(f"target_texts: {target_texts}")
        with torch.no_grad():
            # Image features (仍需每次计算)
            f_I = self.clip_model.encode_image(images)
            f_I = f_I / f_I.norm(dim=1, keepdim=True)

            #  Text features - 直接使用缓存！
            f_T = self.cached_text_features[self.current_variant]

            # CLIP Surgery - 使用缓存的redundant features
            similarity = clips.clip_feature_surgery(
                f_I, f_T, self.cached_redundant_features
            )

            # Find target texts
            similarity1 = (100.0 * f_I[:, 0, :] @ f_T.T).softmax(dim=-1)
            values, indices = similarity1.topk(1, dim=-1)
            indices = indices.squeeze(-1)

            # Get text list based on variant
            if self.current_variant == 'ImageNetS50':
                text_list = all_texts_50
            elif self.current_variant == 'ImageNetS300':
                text_list = all_texts_300
            else:
                text_list = all_texts_919

            target_texts = [text_list[idx.item()] for idx in indices]

        return f_I, f_T, similarity, target_texts

    def forward_point_branch(self, images, text_prompts, f_I, similarity, target_texts):
        """
        Branch 1: Point-Prompt (BATCHED for 3-6x speedup)

        Key optimization: Batch encode all images through SAM at once

        Args:
            images: Input images [B, 3, H, W]
            text_prompts: List of text prompts
            f_I: CLIP image features [B, 1025, 512]
            similarity: Similarity matrix [B, 1025, num_classes]
            target_texts: Target class names for each image [B]

        Returns:
            z: Segmentation masks [B, H, W]
            points: List of points for each image
            labels: List of labels for each image
        """
        import cv2
        import numpy as np
        import torch.nn.functional as F

        batch_size = images.shape[0]
        H, W = images.shape[2], images.shape[3]

        # # Get text list
        # if isinstance(text_prompts, list) and len(text_prompts) > 0:
        #     text_list = text_prompts[0] if isinstance(text_prompts[0], list) else text_prompts
        # else:
        #     text_list = text_prompts
        text_list = text_prompts
        # Stage 1: Prepare images and points
        batch_images_np = []
        batch_point_coords = []
        batch_point_labels = []

        for b in range(batch_size):
            target_idx = text_list.index(target_texts[b])
            sim_map = similarity[b, 1:, target_idx]

            # 使用预计算的参数denormalize (避免重复创建numpy array)
            image_np = images[b].cpu().permute(1, 2, 0).numpy()
            image_np = image_np * self._denorm_std + self._denorm_mean
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
            cv2_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Convert similarity to points
            points, labels = clips.similarity_map_to_points(sim_map, cv2_img.shape[:2], t=0.5)

            point_coords = np.array(points)
            point_labels = np.array(labels)
            positive_mask = point_labels != 0
            point_coords = point_coords[positive_mask]
            point_labels = point_labels[positive_mask]

            if len(point_coords) == 0:
                point_coords = np.array([[H // 2, W // 2]])
                point_labels = np.array([1])

            batch_images_np.append(cv2_img)
            batch_point_coords.append(point_coords)
            batch_point_labels.append(point_labels)

        # Stage 2: Batch SAM encoding
        with torch.no_grad():
            # 批量转换 (更高效)
            batch_images_tensor = torch.stack([
                torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                for img in batch_images_np
            ]).float().to(self.device)

            # Resize to 1024 for SAM
            if batch_images_tensor.shape[2] != 1024 or batch_images_tensor.shape[3] != 1024:
                batch_images_tensor = F.interpolate(
                    batch_images_tensor, size=(1024, 1024),
                    mode='bilinear', align_corners=False
                )

            # Batch encode (主要加速)
            image_embeddings = self.sam_model.image_encoder(batch_images_tensor)

            # Stage 3: Decode per image (fast)
            all_masks = []
            for b in range(batch_size):
                point_coords = batch_point_coords[b]
                point_labels = batch_point_labels[b]

                coords_torch = torch.as_tensor(
                    point_coords, dtype=torch.float, device=self.device
                ).unsqueeze(0)
                labels_torch = torch.as_tensor(
                    point_labels, dtype=torch.int, device=self.device
                ).unsqueeze(0)

                sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                    points=(coords_torch, labels_torch), boxes=None, masks=None
                )

                low_res_masks, iou_predictions,_ = self.sam_model.mask_decoder(
                    image_embeddings=image_embeddings[b:b + 1],
                    image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                masks = F.interpolate(low_res_masks, (H, W), mode="bilinear", align_corners=False)
                mask = (masks > self.sam_model.mask_threshold)[0, 0]
                all_masks.append(mask)

        z = torch.stack(all_masks)
        return z, batch_point_coords, batch_point_labels

        # # ==========================================
        # # Stage 1: Prepare all images and points
        # # ==========================================
        # batch_images_np = []
        # batch_point_coords = []
        # batch_point_labels = []
        #
        # for b in range(batch_size):
        #     # Find target class index
        #     target_text = target_texts[b]
        #     try:
        #         target_idx = text_list.index(target_text)
        #     except ValueError:
        #         target_idx = 0
        #
        #     # Get similarity map for this image and target class
        #     sim_map = similarity[b, 1:, target_idx]  # [1024] - exclude CLS token
        #
        #     # Denormalize image
        #     image_np = images[b].cpu().permute(1, 2, 0).numpy()
        #     image_np = (image_np * np.array([0.26862954, 0.26130258, 0.27577711]) +
        #                 np.array([0.48145466, 0.4578275, 0.40821073]))
        #     image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        #     cv2_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        #
        #     # Convert similarity map to points
        #     points, labels = clips.similarity_map_to_points(
        #         sim_map,
        #         cv2_img.shape[:2],
        #         t=0.5
        #     )
        #
        #     # Filter positive points
        #     point_coords = np.array(points)
        #     point_labels = np.array(labels)
        #     positive_mask = point_labels != 0
        #     point_coords = point_coords[positive_mask]
        #     point_labels = point_labels[positive_mask]
        #
        #     # Handle empty points
        #     if len(point_coords) == 0:
        #         point_coords = np.array([[H // 2, W // 2]])
        #         point_labels = np.array([1])
        #
        #     batch_images_np.append(cv2_img)
        #     batch_point_coords.append(point_coords)
        #     batch_point_labels.append(point_labels)
        #
        # # ==========================================
        # # Stage 2: BATCH encode all images at once
        # # ==========================================
        # all_masks = []
        #
        # with torch.no_grad():
        #     # Convert to tensor format [B, 3, H, W]
        #     batch_images_tensor = []
        #     for img in batch_images_np:
        #         # SAM expects RGB in [0, 255]
        #         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #         batch_images_tensor.append(torch.from_numpy(img_rgb).permute(2, 0, 1))
        #
        #     batch_images_tensor = torch.stack(batch_images_tensor).float().to(self.device)
        #     if batch_images_tensor.shape[2] != 1024 or batch_images_tensor.shape[3] != 1024:
        #         batch_images_tensor = F.interpolate(
        #             batch_images_tensor,
        #             size=(1024, 1024),
        #             mode='bilinear',
        #             align_corners=False
        #         )
        #
        #     # print("batch_images_tensor", batch_images_tensor.shape)#torch.Size([80, 3, 512, 512])
        #
        #     # *** KEY OPTIMIZATION: Batch image encoding ***
        #     # This is the main speedup - encoding all images at once
        #     image_embeddings = self.sam_model.image_encoder(batch_images_tensor)
        #     # [B, 256, 64, 64] - All images encoded in one forward pass!
        #
        #     # ==========================================
        #     # Stage 3: Decode with points for each image
        #     # ==========================================
        #     for b in range(batch_size):
        #         # Get points for this image
        #         point_coords = batch_point_coords[b]
        #         point_labels = batch_point_labels[b]
        #
        #         # Convert to tensor
        #         coords_torch = torch.as_tensor(
        #             point_coords, dtype=torch.float, device=self.device
        #         ).unsqueeze(0)  # [1, N, 2]
        #
        #         labels_torch = torch.as_tensor(
        #             point_labels, dtype=torch.int, device=self.device
        #         ).unsqueeze(0)  # [1, N]
        #
        #         # Encode prompts (fast)
        #         sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
        #             points=(coords_torch, labels_torch),
        #             boxes=None,
        #             masks=None,
        #         )
        #
        #         # Decode using the pre-computed image embedding (fast)
        #         low_res_masks, iou_predictions,_ = self.sam_model.mask_decoder(
        #             image_embeddings=image_embeddings[b:b + 1],  # [1, 256, 64, 64] - Use cached!
        #             image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
        #             sparse_prompt_embeddings=sparse_embeddings,
        #             dense_prompt_embeddings=dense_embeddings,
        #             multimask_output=False,
        #         )
        #
        #         # Upsample to original resolution
        #         masks = F.interpolate(
        #             low_res_masks,
        #             (H, W),
        #             mode="bilinear",
        #             align_corners=False,
        #         )
        #
        #         mask = (masks > self.sam_model.mask_threshold)[0, 0]
        #         all_masks.append(mask)
        #
        # z = torch.stack(all_masks)
        # return z, batch_point_coords, batch_point_labels

    def forward_box_branch(self, images, target_texts):
        """
        Branch 2: Box-Prompt (BATCHED for 3-6x speedup)

        Key optimization: Batch encode all images through SAM at once

        Args:
            images: Input images [B, 3, H, W]
            target_texts: Target class names [B]

        Returns:
            z_hat: Segmentation masks [B, H, W]
            B_init: List of detection boxes
            sam_tokens: List of SAM tokens
        """
        import cv2
        import numpy as np
        from groundingdino.util.inference import predict
        from groundingdino.util import box_ops
        from PIL import Image as PILImage
        from torchvision import transforms
        import torch.nn.functional as F

        batch_size = images.shape[0]
        H, W = images.shape[2], images.shape[3]

        # ==========================================
        # Stage 1: Prepare all images for Grounding DINO
        # ==========================================
        batch_images_np = []
        batch_images_dino = []
        transform_dino = transforms.Compose([transforms.ToTensor()])

        for b in range(batch_size):
            # Denormalize
            image_np = images[b].cpu().permute(1, 2, 0).numpy()
            # image_np = (image_np * np.array([0.26862954, 0.26130258, 0.27577711]) +
            #             np.array([0.48145466, 0.4578275, 0.40821073]))
            image_np = image_np * self._denorm_std + self._denorm_mean
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

            pil_img = PILImage.fromarray(image_np)
            image_dino = transform_dino(pil_img).to(self.device)

            batch_images_np.append(image_np)
            batch_images_dino.append(image_dino)

        # ==========================================
        # Stage 2: Detect boxes with Grounding DINO
        # ==========================================
        all_boxes_list = []

        for b in range(batch_size):
            target_text = target_texts[b]
            image_dino = batch_images_dino[b]

            try:
                boxes, logits, phrases = predict(
                    model=self.grounding_dino,
                    image=image_dino,
                    caption=target_text,
                    box_threshold=0.2,
                    text_threshold=0.2,
                )
            except Exception as e:
                all_boxes_list.append(torch.zeros(0, 4, device=self.device))
                continue

            if len(boxes) == 0:
                all_boxes_list.append(torch.zeros(0, 4, device=self.device))
            else:
                boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor(
                    [W, H, W, H], device=boxes.device
                )
                all_boxes_list.append(boxes_xyxy)

        # ==========================================
        # Stage 3: BATCH encode all images at once
        # ==========================================
        all_masks = []
        all_sam_tokens_list = []

        with torch.no_grad():
            # Convert to tensor format [B, 3, H, W]
            batch_images_tensor = []
            for img_np in batch_images_np:
                # SAM expects RGB in [0, 255]
                batch_images_tensor.append(torch.from_numpy(img_np).permute(2, 0, 1))

            batch_images_tensor = torch.stack(batch_images_tensor).float().to(self.device)
            if batch_images_tensor.shape[2] != 1024 or batch_images_tensor.shape[3] != 1024:
                batch_images_tensor = F.interpolate(
                    batch_images_tensor,
                    size=(1024, 1024),
                    mode='bilinear',
                    align_corners=False
                )

            # *** KEY OPTIMIZATION: Batch image encoding ***
            # This is the main speedup - encoding all images at once
            image_embeddings = self.sam_model.image_encoder(batch_images_tensor)
            # [B, 256, 64, 64] - All images encoded in one forward pass!

            # ==========================================
            # Stage 4: Decode with boxes for each image
            # ==========================================
            for b in range(batch_size):
                boxes_xyxy = all_boxes_list[b]

                if len(boxes_xyxy) == 0:
                    all_masks.append(torch.zeros(H, W, device=self.device))
                    all_sam_tokens_list.append(torch.zeros(0, 256, device=self.device))
                    continue

                # Convert boxes to tensor
                # boxes_tensor = boxes_xyxy.unsqueeze(0) # [1, N, 4]
                boxes_tensor=boxes_xyxy
                # *** ensure device once again ***
                if boxes_tensor.device != self.device:
                    boxes_tensor = boxes_tensor.to(self.device)
                # print("box_tensor:",boxes_tensor.shape)#[5,4]
                # Encode prompts (fast)
                sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                    points=None,
                    boxes=boxes_tensor,
                    masks=None,
                )

                # Decode using the pre-computed image embedding (fast)
                low_res_masks, iou_predictions,sam_tokens= self.sam_model.mask_decoder(
                    image_embeddings=image_embeddings[b:b + 1],  # [1, 256, 64, 64] - Use cached!
                    image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                # Upsample to original resolution
                masks = F.interpolate(
                    low_res_masks,
                    (H, W),
                    mode="bilinear",
                    align_corners=False,
                )

                # Combine multi-box masks
                mask = (masks > self.sam_model.mask_threshold).sum(dim=0)[0] > 0

                all_masks.append(mask)
                all_sam_tokens_list.append(sam_tokens)

        z_hat = torch.stack(all_masks)
        return z_hat, all_boxes_list, all_sam_tokens_list

    def get_clip_feature_map(self, images, f_I):
        """Get CLIP feature map for RoIAlign"""
        B, C = f_I.shape[0], f_I.shape[2]
        patch_features = f_I[:, 1:, :]  # [B, 1024, 512]
        H_feat = W_feat = 32
        feature_map = patch_features.reshape(B, H_feat, W_feat, C)
        feature_map = feature_map.permute(0, 3, 1, 2)  # [B, 512, 32, 32]
        return feature_map

