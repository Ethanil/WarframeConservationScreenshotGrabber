import os
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import csv
import re
import concurrent.futures
from datetime import datetime

testmode = False
showimages = False


# Crop the image with the parameters and use image_to_data from tesseract to get text
def get_slice(image, tesseract_config_of_the_slice: str, top_position: float, slice_thickness: float,
              left_position=0.0, right_position=1.0):
    result = []
    # crop the image
    image_height, image_width = image.shape[:2]
    image = image[int(image_height * top_position):int(image_height * (top_position + slice_thickness)),
                  int(image_width * left_position):int(image_width * right_position)]
    # read the value
    image_data = pytesseract.image_to_data(image, output_type=Output.DICT, config=tesseract_config_of_the_slice)
    image_data_length = len(image_data['text'])
    for data_entry in range(image_data_length):
        if int(float(image_data['conf'][data_entry])) > 60:
            result.append(image_data['text'][data_entry])
    return result, image


# difference between input image and Opening of the image
def tophat(image, kernel_size):
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                               (kernel_size, kernel_size)))


# Cleans the given list-input and checks it against the ANIMAL_LIST constant and returns the cleaned string
def get_name(name_input):
    for word in name_input:
        if word.upper() not in PREFIX_LIST:
            name_input.remove(word)
    if len(name_input) == 0:
        return None
    joined_string = ' '.join(name_input)
    if joined_string.upper() in ANIMAL_LIST:
        return joined_string
    else:
        return None


# Checks if the given list is a potential age-input and returns the age as a dictionary
def get_age(age_input):
    if len(age_input) == 0 or len(age_input) % 2 != 0:
        return None
    age_statements = {}
    for x in range(0, len(age_input), 2):
        if not age_input[x].isdigit():
            return None
        if age_input[x+1] not in AGE_LIST:
            return None
        if re.search(age_input[x+1], r'(year|years)'):
            age_statements['years'] = int(age_input[x])
        if re.search(age_input[x+1], r'(month|months)'):
            age_statements['months'] = int(age_input[x])
    return age_statements


# Checks if the given list is a potential weight-input and returns the weight as a float
def get_weight(weight_input):
    if len(weight_input) == 0:
        return None
    if len(weight_input) == 2 and weight_input[1].lower() == 'kg' \
            and re.search(r'^\d+\.\d+$', weight_input[0]) is not None:
        return float(weight_input[0])
    elif len(weight_input) == 1 and re.search(r'^\d+\.\dkg+$', weight_input[0]) is not None:
        return float(weight_input[0][:-2])
    else:
        return None


# Checks if the given list is a potential gender-input and returns the gender as a string
def get_gender(gender_input):
    gen_input = gender_input.copy()
    for word in gen_input:
        if word not in GENDER_LIST:
            gen_input.remove(word)
    if len(gen_input) == 0:
        return None
    joined_string = ''.join(gen_input)
    if joined_string in GENDER_LIST:
        return joined_string
    else:
        return None


# Checks if the given list is a potential note-input and returns the note as a string
def get_note(note_input):
    if len(note_input) == 0:
        return None
    joined_string = ' '.join(note_input)
    if joined_string.lower() not in QUIRK_LIST:
        return None
    else:
        return joined_string


# creates a filepath to the directory in which the animal should be stored
def get_save_location(screenshot_directory, animalname='', get_parent_directory=False):
    # separate the species name from the subspecies
    animalname_list = animalname.split()
    only_species = ''
    only_subspecies = ''
    if len(animalname_list) == 2:
        if animalname_list[0] in ['HORRASQUE', 'KUBRODON']:
            only_species = '\\' + animalname_list.pop(0)
            only_subspecies = '\\' + animalname_list.pop(0)
        else:
            only_subspecies = '\\' + animalname_list.pop(0)
            only_species = '\\' + animalname_list.pop(0)
    elif len(animalname_list) == 3:
        if animalname_list[2] == 'KAVAT':
            only_subspecies = '\\' + animalname_list.pop(0)
            only_species = '\\' + '_'.join(animalname_list)
        elif animalname_list[2] in ['STOVER', 'SAWGAW']:
            only_species = '\\' + animalname_list.pop(2)
            only_subspecies = '\\' + '_'.join(animalname_list)
    else:
        only_species = '\\' + 'ERROR'
        only_subspecies = ''

    # create a dir for that subspecies, if it does not exist already
    parent_directory = os.path.dirname(screenshot_directory)+'\\Warframe_Animals'
    if not os.path.isdir(parent_directory):
        os.mkdir(parent_directory)
    parent_directory = parent_directory+'\\Animals'
    if not os.path.isdir(parent_directory):
        os.mkdir(parent_directory)
    if get_parent_directory:
        return parent_directory
    if not os.path.isdir(parent_directory + only_species):
        os.mkdir(parent_directory + only_species)
    if not os.path.isdir(parent_directory + only_species + only_subspecies):
        os.mkdir(parent_directory + only_species + only_subspecies)
    return parent_directory + only_species + only_subspecies


# tries to find the name of the animal in the given image and returns it as a string
def find_name(image, config):
    height, width = image.shape[:2]
    kernelsize = 9
    name = None
    while not name and kernelsize < 15:
        img_animalname = image[int(height * 0.08):int(height * 0.12), int(width * 0.29):int(width * 0.74)]
        img_animalname = tophat(img_animalname, kernelsize)
        kernelsize += 3
        for thickness in np.arange(1, 0, -0.01):
            for start_position in np.arange(0, 1 - thickness, 0.01):
                slice_string, slice_image = get_slice(img_animalname, config, start_position, thickness)
                name = get_name(slice_string)
                if testmode:
                    print(name)
                    if showimages:
                        cv2.imshow('img_infobox_full', slice_image)
                        cv2.waitKey(0)
                if name:
                    return name
    return name


# searches the given infobox image for information and returns them
def find_infobox_contents(image, config, name):
    # get the correct size of the Infobox based on the animal-name, if no name was found, use huge proportions to gather
    # as much data as possible
    if name:
        infobox_top_start = INFOBOX_POSITION_LIST[name][0]
        infobox_bottom_end = infobox_top_start + INFOBOX_POSITION_LIST[name][1]
        if testmode:
            print('found {} in INFOBOX_POSITION_LIST using {} as start and {} as endpoint'.format(name,
                                                                                                  infobox_top_start,
                                                                                                  infobox_bottom_end))
    else:
        infobox_top_start = 0.52
        infobox_bottom_end = infobox_top_start + 0.31
    infobox_height = infobox_bottom_end - infobox_top_start
    height, width = image.shape[:2]
    years = None
    months = None
    weight = None
    gender = None
    note = None
    loop = 0
    while not(years and months and weight and gender and note) and loop < 12:
        # crop the image to the right size to evaluate the infobox
        img_infobox_plain = image[int(height * infobox_top_start):int(height * infobox_bottom_end),
                                  int(width * 0.75):int(width * 0.91)]
        img_infobox_full = img_infobox_plain.copy()
        if loop < 8:
            img_infobox_full = tophat(img_infobox_full, 8 - loop)
        else:
            img_infobox_full = tophat(img_infobox_full, 16 - loop)
            img_infobox_full = cv2.cvtColor(img_infobox_full, cv2.COLOR_BGR2GRAY)
            img_infobox_full = cv2.Canny(img_infobox_full, 30, 100)
        loop += 1

        if testmode and showimages:
            cv2.imshow('img_infobox_full', img_infobox_full)
        thickness = 0.028 / infobox_height
        # while thickness < 0.035 / infobox_height:
        position = 0
        endposition = 1-thickness
        for skip in (years, weight, gender, note):
            if skip is not None:
                if testmode:
                    print('skipping a line')
                position += thickness
            else:
                break
        for skip in (note, gender, weight, years):
            if skip is not None:
                if testmode:
                    print('skipping a line')
                endposition -= thickness
            else:
                break
        while position < endposition:
            position_step = 0.01
            slice_string, slice_image = get_slice(img_infobox_full, config, position, thickness)
            if testmode:
                print(slice_string)
                if showimages:
                    cv2.imshow('slice_image', slice_image)
            age_temp = get_age(slice_string)
            weight_temp = get_weight(slice_string)
            if name in HAS_NO_GENDER_LIST:
                gender = 'Genderless'
                gender_temp = None
            else:
                gender_temp = get_gender(slice_string)
            if name in HAS_NO_QUIRKS_LIST:
                note = 'No Quirk'
                note_temp = None
            else:
                note_temp = get_note(slice_string)

            if age_temp is not None and ((years is None or months is None) or len(saved_age) < len(age_temp)):
                saved_age = age_temp
                if len(age_temp) > 1:
                    if testmode:
                        print('skipping a line')
                    position_step += 0.8 * thickness
                if years is None or years < age_temp.get('years', 0):
                    if testmode:
                        print('found new years value: {}; old: {}'.format(age_temp.get('years', 0), years))
                    years = age_temp.get('years', 0)
                if months is None or months < age_temp.get('months', 0):
                    if testmode:
                        print('found new months value: {}; old: {}'.format(age_temp.get('months', 0), months))
                    months = age_temp.get('months', 0)
            if weight_temp is not None and weight is None:
                if testmode:
                    print('skipping a line')
                position_step += 0.8 * thickness
                if name in HAS_NO_GENDER_LIST:
                    if testmode:
                        print('skipping a line')
                    position_step += 0.8 * thickness
                if testmode:
                    print('found new weight value: {}; old: {}'.format(weight_temp, weight))
                weight = weight_temp
            if gender_temp is not None and gender is None:
                if name in HAS_NO_QUIRKS_LIST:
                    if testmode:
                        print('skipping a line')
                    position_step += 0.8 * thickness
                if testmode:
                    print('skipping a line')
                position_step += 0.8 * thickness
                if testmode:
                    print('found new gender value: {}; old: {}'.format(gender_temp, gender))
                gender = gender_temp
            if note_temp is not None and note is None:
                if testmode:
                    print('skipping a line')
                position_step += 0.8 * thickness
                if testmode:
                    print('found new note value: {}; old: {}'.format(note_temp, note))
                note = note_temp
            position += position_step
            if testmode:
                print('Year: {} Month: {} Weight: {} Gender: {} Note {}'
                      .format(years, months, weight, gender, note))
                if showimages:
                    cv2.waitKey(0)
            if None not in (years, months, weight, gender, note):
                return years, months, weight, gender, note
        # thickness += 0.01
    return years, months, weight, gender, note


# the main function to do all the work
def create_data(filepath, par_directory):
    # define Variables
    error_msg = []
    tesseract_config = '-l eng --psm 7'

    # read image
    print('Starting to process {}'.format(filepath))
    img = cv2.imread(filepath)

    # recolor input for better readability for tesseract
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # find the name of the animal
    name = find_name(img, tesseract_config)

    years, months, weight, gender, note = find_infobox_contents(img, tesseract_config, name)

    if None not in (years, months, weight, gender, note):
        all_data_correct = True
    else:
        all_data_correct = False
        if name is None:
            error_msg.append('name')
        if years is None:
            error_msg.append('years')
        if months is None:
            error_msg.append('months')
        if weight is None:
            error_msg.append('weight')
        if gender is None:
            error_msg.append('gender')
        if note is None:
            error_msg.append('note')

    if testmode:
        print([name, years, months, weight, gender, note, os.path.basename(filepath), ''])
    if all_data_correct:
        future_dir = get_save_location(par_directory, name)+'\\'+os.path.basename(filepath)
        if not testmode:
            os.rename(filepath, future_dir)
        print('Moved File to ' + future_dir)
        return [name, years, months, weight, gender, note, os.path.basename(filepath), '']
    else:
        future_dir = get_save_location(par_directory, '')+'\\'+os.path.basename(filepath)
        if not testmode:
            os.rename(filepath, future_dir)
        print('Something was wrong with: {} manual input needed\nMoved File to {}'.format(os.path.basename(filepath),
                                                                                          future_dir))
        return [name, years, months, weight, gender, note, os.path.basename(filepath),
                error_msg]


# lists to check the correctness of the found input
INFOBOX_POSITION_LIST = \
   {'BURROWING CRYPTILEX':    [0.65, 0.13], 'SEPTIC CRYPTILEX':      [0.68, 0.13], 'CAUSTIC CRYPTILEX':    [0.64, 0.14],
    'COMMON AVICHAEA':        [0.67, 0.14], 'SPORULE AVICHAEA':      [0.57, 0.14], 'VISCID AVICHAEA':      [0.57, 0.14],
    'AMETHYST NEXIFERA':      [0.64, 0.14], 'SCARLET NEXIFERA':      [0.65, 0.14], 'VIRIDIAN NEXIFERA':    [0.65, 0.14],
    'VIZIER PREDASITE':       [0.65, 0.13], 'PHARAOH PREDASITE':     [0.57, 0.14], 'MEDJAY PREDASITE':     [0.57, 0.14],
    'SLY VULPAPHYLA':         [0.65, 0.14], 'CRESCENT VULPAPHYLA':   [0.55, 0.13], 'PANZER VULPAPHYLA':    [0.57, 0.14],
    'UMBER UNDAZOA':          [0.65, 0.14], 'HOWLER UNDAZOA':        [0.55, 0.14], 'VAPOROUS UNDAZOA':     [0.57, 0.14],
    'GREEN VELOCIPOD':        [0.67, 0.12], 'PURPLE VELOCIPOD':      [0.67, 0.12], 'WHITE VELOCIPOD':      [0.70, 0.10],
    'PLAINS KUAKA':           [0.67, 0.14], 'ASHEN KUAKA':           [0.67, 0.14], 'GHOST KUAKA':          [0.67, 0.14],
    'COMMON CONDROC':         [0.57, 0.14], 'ROGUE CONDROC':         [0.57, 0.14], 'EMPEROR CONDROC':      [0.57, 0.14],
    'COASTAL MERGOO':         [0.52, 0.14], 'WOODLAND MERGOO':       [0.52, 0.14], 'SPLENDID MERGOO':      [0.52, 0.14],
    'OSTIA VASCA KAVAT':      [0.44, 0.14], 'BAU VASCA KAVAT':       [0.44, 0.14], 'NEPHIL VASCA KAVAT':   [0.44, 0.14],
    'SUNNY POBBER':           [0.54, 0.14], 'DELICATE POBBER':       [0.54, 0.14], 'SUBTERRANEAN POBBER':  [0.54, 0.14],
    'WHITE-BREASTED VIRMINK': [0.57, 0.14], 'DUSKY-HEADED VIRMINK':  [0.57, 0.14], 'RED-CRESTED VIRMINK':  [0.57, 0.14],
    'FLOSSY SAWGAW':          [0.62, 0.14], 'ALPINE MONITOR SAWGAW': [0.62, 0.14], 'FROGMOUTHED SAWGAW':   [0.62, 0.14],
    'SPOTTED BOLAROLA':       [0.52, 0.31], 'BLACK-BANDED BOLAROLA': [0.52, 0.31], 'THORNY BOLAROLA':      [0.52, 0.31],
    'DAPPLED HORRASQUE':      [0.55, 0.13], 'SWIMMER HORRASQUE':     [0.55, 0.13], 'HORRASQUE STORMER':    [0.55, 0.13],
    'SENTINEL STOVER':        [0.62, 0.14], 'FUMING DAX STOVER':     [0.62, 0.14], 'FIRE-VEINED STOVER':   [0.62, 0.14],
    'BRINDLE KUBRODON':       [0.57, 0.14], 'VALLIS KUBRODON':       [0.57, 0.14], 'KUBRODON INCARNADINE': [0.57, 0.14]}
HAS_NO_QUIRKS_LIST = ['GREEN VELOCIPOD', 'PURPLE VELOCIPOD', 'WHITE VELOCIPOD']
HAS_NO_GENDER_LIST = ['GREEN VELOCIPOD', 'PURPLE VELOCIPOD', 'WHITE VELOCIPOD',
                      'BURROWING CRYPTILEX', 'SEPTIC CRYPTILEX', 'CAUSTIC CRYPTILEX']
GENDER_LIST = ['Male', 'Female', 'Genderless']
QUIRK_LIST = ['abnormal swelling', 'acute hearing', 'afraid of heights', 'afraid of water', 'alert', 'always hiding',
              'bad breath', 'big eyes', 'binge-eater', 'bites nails', 'burrows under blankets', 'calm',
              'can open closets', 'chest infection', 'chews wires', 'chirps', 'collects flowers', 'constipated',
              'cracked footpads', 'cracked tooth', 'damaged beak', 'defensive', 'digger', 'dirty', 'discolored teeth',
              'double-jointed', 'dry skin', 'dull, waxy feathers', 'easily startled', 'enlarged genitals',
              'excessive grooming', 'fever', 'fleas', 'freezes when panicked', 'fussy eater', 'grouchy',
              'healthy and happy', 'highly aggressive', 'in heat', 'intestinal parasites', 'irritable', 'itchy',
              'joint stiffness', 'lesions', 'lethargic', 'likes being carried', 'likes open space', 'likes people',
              'lively', 'lolling tongue', 'loves water', 'mange', 'muscular', 'occasional discharge', 'odd',
              'overweight', 'physically undeveloped', 'playful', 'pleasant smelling', 'poor vision',
              'prefers confined spaces', 'prefers high places', 'pronounced limp', 'protective', 'purrs',
              'recognizes faces', 'redness', 'rheumy eyes', 'runs in circles', 'scavenges relentlessly',
              'scratches the ground', 'shivers inexplicably', 'sings', 'smelly', 'snarls', 'sneaky', 'starving',
              'strange habits', 'territorial', 'torn ear', 'twitchy', 'underweight', 'whistles', 'drools', 'keen smell',
              'binge eater', 'preens', 'ear infection', 'loves heights', 'undeveloped teeth', 'clean', 'old injuries',
              'molting', 'shiny feathers', 'bites claws', 'mites', 'healthy', 'cracked claws', 'tremors',
              'pecks things', 'waxy ears', 'licks things', 'cracked teeth', 'misaligned beak', 'highly active']
AGE_LIST = {'year', 'years', 'month', 'months'}
SPECIES_LIST = ['CRYPTILEX', 'AVICHAEA', 'NEXIFERA', 'PREDASITE', 'VULPAPHYLA', 'VELOCIPOD', 'UNDAZOA', 'KUAKA',
                'CONDROC', 'MERGOO', 'VASCA KAVAT', 'POBBER', 'VIRMINK', 'SAWGAW', 'BOLAROLA', 'HORRASQUE', 'STOVER',
                'KUBRODON']
PREFIX_LIST = ['CRYPTILEX', 'AVICHAEA', 'NEXIFERA', 'PREDASITE', 'VULPAPHYLA', 'VELOCIPOD', 'UNDAZOA', 'KUAKA',
               'CONDROC', 'MERGOO', 'VASCA', 'KAVAT', 'POBBER', 'VIRMINK', 'SAWGAW', 'BOLAROLA', 'HORRASQUE', 'STOVER',
               'KUBRODON',
               'BURROWING', 'SEPTIC', 'CAUSTIC',
               'COMMON', 'SPORULE', 'VISCID',
               'AMETHYST', 'SCARLET', 'VIRIDIAN',
               'VIZIER', 'PHARAOH', 'MEDJAY',
               'SLY', 'CRESCENT', 'PANZER',
               'UMBER', 'HOWLER', 'VAPOROUS',
               'GREEN', 'PURPLE', 'WHITE',
               'PLAINS', 'ASHEN', 'GHOST',
               'COMMON', 'ROGUE', 'EMPEROR',
               'COASTAL', 'WOODLAND', 'SPLENDID',
               'OSTIA', 'BAU', 'NEPHIL',
               'SUNNY', 'DELICATE', 'SUBTERRANEAN',
               'WHITE-BREASTED', 'DUSKY-HEADED', 'RED-CRESTED',
               'FLOSSY', 'ALPINE', 'MONITOR', 'FROGMOUTHED',
               'SPOTTED', 'BLACK-BANDED', 'THORNY',
               'DAPPLED', 'SWIMMER', 'STORMER',
               'SENTINEL', 'FUMING DAX', 'FIRE-VEINED',
               'BRINDLE', 'VALLIS', 'INCARNADINE']
ANIMAL_LIST = ['BURROWING CRYPTILEX', 'SEPTIC CRYPTILEX', 'CAUSTIC CRYPTILEX',
               'COMMON AVICHAEA', 'SPORULE AVICHAEA', 'VISCID AVICHAEA',
               'AMETHYST NEXIFERA', 'SCARLET NEXIFERA', 'VIRIDIAN NEXIFERA',
               'VIZIER PREDASITE', 'PHARAOH PREDASITE', 'MEDJAY PREDASITE',
               'SLY VULPAPHYLA', 'CRESCENT VULPAPHYLA', 'PANZER VULPAPHYLA',
               'UMBER UNDAZOA', 'HOWLER UNDAZOA', 'VAPOROUS UNDAZOA',
               'GREEN VELOCIPOD', 'PURPLE VELOCIPOD', 'WHITE VELOCIPOD',
               'PLAINS KUAKA', 'ASHEN KUAKA', 'GHOST KUAKA',
               'COMMON CONDROC', 'ROGUE CONDROC', 'EMPEROR CONDROC',
               'COASTAL MERGOO', 'WOODLAND MERGOO', 'SPLENDID MERGOO',
               'OSTIA VASCA KAVAT', 'BAU VASCA KAVAT', 'NEPHIL VASCA KAVAT',
               'SUNNY POBBER', 'DELICATE POBBER', 'SUBTERRANEAN POBBER',
               'WHITE-BREASTED VIRMINK', 'DUSKY-HEADED VIRMINK', 'RED-CRESTED VIRMINK',
               'FLOSSY SAWGAW', 'ALPINE MONITOR SAWGAW', 'FROGMOUTHED SAWGAW',
               'SPOTTED BOLAROLA', 'BLACK-BANDED BOLAROLA', 'THORNY BOLAROLA',
               'DAPPLED HORRASQUE', 'SWIMMER HORRASQUE', 'HORRASQUE STORMER',
               'SENTINEL STOVER', 'FUMING DAX STOVER', 'FIRE-VEINED STOVER',
               'BRINDLE KUBRODON', 'VALLIS KUBRODON', 'KUBRODON INCARNADINE']

header = ['name', 'years', 'months', 'weight', 'gender', 'quirk', 'filename', 'error?']
csv_data = []
error_data = []
directory = 'D:\\Conservation\\Screenshots'
files_to_check = []


# find all the files in the directory that are .jpg and add them to files_to_check
for root, dirs, files in os.walk(directory, topdown=False):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.png')):
            files_to_check.append(os.path.join(root, filename))
            if testmode:
                data = create_data(os.path.join(root, filename), directory)
                if data[7] == '':
                    csv_data.append(data)
                else:
                    error_data.append(data)

if not testmode:
    critical_errors = 0
    # save the start-time to give an estimate on how long the operation took
    start_time = datetime.now()
    # Go through all images with multithreading:
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        print('{}: Starting to process {} files'.format(start_time.strftime("%H:%M:%S"), len(files_to_check)))
        future_to_data = {executor.submit(create_data, filepath, directory): filepath for filepath in files_to_check}
        counter = 0
        for future in concurrent.futures.as_completed(future_to_data):
            counter += 1
            filepath = future_to_data[future]
            try:
                data = future.result()
            except Exception as exc:
                print('{} generated an exception: {}'.format(filepath, exc))
                critical_errors += 1
            else:
                print('\nFinished processing {}\n{}/{} done'.format(filepath, counter, len(files_to_check)))
                if data[7] == '':
                    csv_data.append(data)
                else:
                    error_data.append(data)
    csv_file = get_save_location(directory, '', True)+'\\animals.csv'
    if os.path.isfile(csv_file):
        print('opening csv file: '+csv_file)
        with open(csv_file, mode='a', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the data
            writer.writerows(csv_data)
    else:
        print('creating csv file: '+csv_file)
        with open(csv_file, mode='a', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write the data
            writer.writerows(csv_data)
    print('wrote {} lines of data into {}'.format(len(csv_data), csv_file))

    csv_file = get_save_location(directory, '', True)+'\\error.csv'
    if os.path.isfile(csv_file):
        print('opening csv file: '+csv_file)
        with open(csv_file, mode='a', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the data
            writer.writerows(error_data)
    else:
        print('creating csv file: '+csv_file)
        with open(csv_file, mode='a', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write the data
            writer.writerows(error_data)
    if len(error_data) > 0:
        print('Encountered {} problems and wrote the data into {}'.format(len(error_data), csv_file))
    if critical_errors > 0:
        print('Encountered {} critical errors; these files will remain in the input folder'.format(critical_errors))
    end_time = datetime.now()
    lapsed_time = end_time-start_time
    duration_in_s = lapsed_time.total_seconds()
    print('It took {} hours {} minutes and {} seconds to scan {} images'.format(int(divmod(duration_in_s, 3600)[0]),
                                                                                int(divmod(duration_in_s, 60)[0]),
                                                                                duration_in_s, len(files_to_check)))
    print('It took {} seconds on average to scan a single image'.format(float(duration_in_s / len(files_to_check))))
