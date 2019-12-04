from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image
import os
from midiutil.MidiFile import MIDIFile

from Segmentation import Segmenter
from template_definition import *
from utils import locate_templates, merge_boxes, collect_info, Bar, Staff

if __name__ == "__main__":

    input_path = "resources/samples/"
    output_root = "output/"
    input_file = "mary.jpg"
    filename = input_file.split('.')[-2]

    img_file = input_path + input_file
    output_path = output_root + filename + '/'
    output_log = output_path + 'log.txt'
    os.makedirs(os.path.dirname(output_log), exist_ok = True)

    segmenter = Segmenter("resources/samples/1.jpeg")
    # return patch contain a line
    staffs = []
    for patch in segmenter.segment():
        staffs.append(patch)
        # break

    print("Segmentation Finished!")

    #-------------------------------------------------------------------------------
    # Symbol Segmentation, Object Recognition, and Semantic Reconstruction
    #-------------------------------------------------------------------------------


    staff_imgs_color = []

    for i in range(len(staffs)):
        red = (255, 0, 0)
        box_thickness = 2
        staff_img = staffs[i].getImage()
        staff_img_color = staff_img.copy()
        staff_img_color = cv2.cvtColor(staff_img_color, cv2.COLOR_GRAY2RGB)

        # ------- Clef -------
        for clef in clef_imgs:
            print("[INFO] Matching {} clef template on staff".format(clef), i + 1)
            clef_boxes = locate_templates(staff_img, clef_imgs[clef], clef_lower, clef_upper, clef_thresh)
            clef_boxes = merge_boxes([j for i in clef_boxes for j in i], 0.3)

            if (len(clef_boxes) == 1):
                print("[INFO] Clef Found: ", clef)
                staffs[i].setClef(clef)

                print("[INFO] Displaying Matching Results on staff", i + 1)
                clef_boxes_img = staffs[i].getImage()
                clef_boxes_img = clef_boxes_img.copy()

                for boxes in clef_boxes:
                    boxes.draw(staff_img_color, red, box_thickness)
                    x = int(boxes.getCorner()[0] + (boxes.getWidth() // 2))
                    y = int(boxes.getCorner()[1] + boxes.getHeight() + 10)
                    cv2.putText(staff_img_color, "{} clef".format(clef), (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.9, red)
                break

        else:
            # A clef should always be found
            print("[INFO] No clef found on staff", i+1)

        # # ------- Time -------
        for time in time_imgs:
            print("[INFO] Matching {} time signature template on staff".format(time), i + 1)
            time_boxes = locate_templates(staff_img, time_imgs[time], time_lower, time_upper, time_thresh)
            time_boxes = merge_boxes([j for i in time_boxes for j in i], 0.5)

            if (len(time_boxes) == 1):
                print("[INFO] Time Signature Found: ", time)
                staffs[i].setTimeSignature(time)

                print("[INFO] Displaying Matching Results on staff", i + 1)

                for boxes in time_boxes:
                    boxes.draw(staff_img_color, red, box_thickness)
                    x = int(boxes.getCorner()[0] - (boxes.getWidth() // 2))
                    y = int(boxes.getCorner()[1] + boxes.getHeight() + 20)
                    cv2.putText(staff_img_color, "{} T".format(time), (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.9, red)
                break

            elif (len(time_boxes) == 0 and i > 0):
                # Take time signature of previous staff
                previousTime = staffs[i-1].getTimeSignature()
                staffs[i].setTimeSignature(previousTime)
                print("[INFO] No time signature found on staff", i + 1, ". Using time signature from previous staff line: ", previousTime)
                break
        else:
            print("[INFO] No time signature available for staff", i + 1)

        staff_imgs_color.append(staff_img_color)


    # ============ Find Primitives ============

    for i in range(len(staffs)):
        print("[INFO] Finding Primitives on Staff ", i+1)
        staff_primitives = []
        staff_img = staffs[i].getImage()
        # rgb image of the staff
        staff_img_color = staff_imgs_color[i].copy()
        box_color = (0, 255, 0)
        box_thickness = 2
        text_color = (255, 0, 0)
        
        # ------- Find primitives on staff -------
        print("[INFO] Matching sharp accidental template...")
        # plt.imshow(staff_img)
        # plt.show()
        sharp_boxes = locate_templates(staff_img, sharp_imgs, sharp_lower, sharp_upper, sharp_thresh)
        sharp_boxes = merge_boxes([j for i in sharp_boxes for j in i], 0.5)
        print("[INFO] Displaying Matching Results on staff", i + 1)
        collect_info(staff_primitives, staff_img_color, box_color, box_thickness, 'sharp', 'S', text_color, sharp_boxes)

        
        print("[INFO] Matching flat accidental template...")
        flat_boxes = locate_templates(staff_img, flat_imgs, flat_lower, flat_upper, flat_thresh)
        flat_boxes = merge_boxes([j for i in flat_boxes for j in i], 0.5)
        print("[INFO] Displaying Matching Results on staff", i + 1)
        collect_info(staff_primitives, staff_img_color, box_color, box_thickness, 'flat', 'D', text_color, flat_boxes)

        
        print("[INFO] Matching quarter note template...")
        quarter_boxes = locate_templates(staff_img, quarter_note_imgs, quarter_note_lower, quarter_note_upper, quarter_note_thresh)
        quarter_boxes = merge_boxes([j for i in quarter_boxes for j in i], 0.5)
        print("[INFO] Displaying Matching Results on staff", i + 1)
        collect_info(staff_primitives, staff_img_color, box_color, box_thickness, 'note', '1/4 N', text_color, quarter_boxes, duration = 1, isNote = True, staff = staffs[i])

        
        print("[INFO] Matching half note template...")
        half_boxes = locate_templates(staff_img, half_note_imgs, half_note_lower, half_note_upper, half_note_thresh)
        half_boxes = merge_boxes([j for i in half_boxes for j in i], 0.5)

        print("[INFO] Displaying Matching Results on staff", i + 1)
        collect_info(staff_primitives, staff_img_color, box_color, box_thickness, 'note', '1/2 N', text_color, half_boxes, duration = 2, isNote = True, staff = staffs[i])
        

        print("[INFO] Matching whole note template...")
        whole_boxes = locate_templates(staff_img, whole_note_imgs, whole_note_lower, whole_note_upper, whole_note_thresh)
        whole_boxes = merge_boxes([j for i in whole_boxes for j in i], 0.5)
        print("[INFO] Displaying Matching Results on staff", i + 1)
        collect_info(staff_primitives, staff_img_color, box_color, box_thickness, 'note', 'N', text_color, whole_boxes, duration = 4, isNote = True, staff = staffs[i])

        
        print("[INFO] Matching eighth rest template...")
        eighth_boxes = locate_templates(staff_img, eighth_rest_imgs, eighth_rest_lower, eighth_rest_upper, eighth_rest_thresh)
        eighth_boxes = merge_boxes([j for i in eighth_boxes for j in i], 0.5)
        print("[INFO] Displaying Matching Results on staff", i + 1)
        collect_info(staff_primitives, staff_img_color, box_color, box_thickness, 'rest', '1/8 R', text_color, eighth_boxes, duration = 0.5)
        
        
        print("[INFO] Matching quarter rest template...")
        quarter_boxes = locate_templates(staff_img, quarter_rest_imgs, quarter_rest_lower, quarter_rest_upper, quarter_rest_thresh)
        quarter_boxes = merge_boxes([j for i in quarter_boxes for j in i], 0.5)
        print("[INFO] Displaying Matching Results on staff", i + 1)
        collect_info(staff_primitives, staff_img_color, box_color, box_thickness, 'rest', '1/4 R', text_color, quarter_boxes, duration = 1)

        
        print("[INFO] Matching half rest template...")
        half_boxes = locate_templates(staff_img, half_rest_imgs, half_rest_lower, half_rest_upper, half_rest_thresh)
        half_boxes = merge_boxes([j for i in half_boxes for j in i], 0.5)
        print("[INFO] Displaying Matching Results on staff", i + 1)
        collect_info(staff_primitives, staff_img_color, box_color, box_thickness, 'rest', '1/2 R', text_color, half_boxes, duration = 2)

        
        print("[INFO] Matching whole rest template...")
        whole_boxes = locate_templates(staff_img, whole_rest_imgs, whole_rest_lower, whole_rest_upper, whole_rest_thresh)
        whole_boxes = merge_boxes([j for i in whole_boxes for j in i], 0.5)
        print("[INFO] Displaying Matching Results on staff", i + 1)
        collect_info(staff_primitives, staff_img_color, box_color, box_thickness, 'rest', 'R', text_color, whole_boxes, duration = 4)

        
        print("[INFO] Matching eighth flag template...")
        flag_boxes = locate_templates(staff_img, eighth_flag_imgs, eighth_flag_lower, eighth_flag_upper, eighth_flag_thresh)
        flag_boxes = merge_boxes([j for i in flag_boxes for j in i], 0.5)
        print("[INFO] Displaying Matching Results on staff", i + 1)
        collect_info(staff_primitives, staff_img_color, box_color, box_thickness, 'eighth_flag', '1/8 f', text_color, flag_boxes, duration = 0)


        print("[INFO] Matching bar line template...")
        bar_boxes = locate_templates(staff_img, bar_imgs, bar_lower, bar_upper, bar_thresh)
        bar_boxes = merge_boxes([j for i in bar_boxes for j in i], 0.5)
        print("[INFO] Displaying Matching Results on staff", i + 1)
        collect_info(staff_primitives, staff_img_color, box_color, box_thickness, 'line', 'bar', text_color, bar_boxes, duration = 0)
        

        print("[INFO] Saving detected primitives in staff {} onto disk".format(i+1))
        cv2.imwrite(output_path + "staff_{}_primitives.jpg".format(i+1), staff_img_color)
        # open_file("output/staff_{}_primitives.jpg".format(i+1))
        
        staff_primitives.sort(key=lambda primitive: primitive.getBox().getCenter())
        print("[INFO] Staff primitives sorted in time")
        eighth_flag_indices = []
        for j in range(len(staff_primitives)):

            if (staff_primitives[j].getPrimitive() == "eighth_flag"):
                # Find all eighth flags
                eighth_flag_indices.append(j)

            if (staff_primitives[j].getPrimitive() == "note"):
                print(staff_primitives[j].getPitch(), end=", ")
            else:
                print(staff_primitives[j].getPrimitive(), end=", ")

        print("\n")
        
        # ------- Correct for eighth notes -------
        # ------- Correct for eighth notes -------
        print("[INFO] Correcting for misclassified eighth notes")
        # Sort out eighth flags
        # Assign to closest note
        for j in eighth_flag_indices:

            distances = []
            distance = staff_primitives[j].getBox().distance(staff_primitives[j-1].getBox())
            distances.append(distance)
            if (j + 1 < len(staff_primitives)):
                distance = staff_primitives[j].getBox().distance(staff_primitives[j+1].getBox())
                distances.append(distance)

            if (distances[1] and distances[0] > distances[1]):
                staff_primitives[j+1].setDuration(0.5)
            else:
                staff_primitives[j-1].setDuration(0.5)

            print("[INFO] Primitive {} was a eighth note misclassified as a quarter note".format(j+1))
            staff_primitives[j] = None

        staff_primitives = [ele for ele in staff_primitives if ele is not None]
        
        # Correct for beamed eighth notes
        # If number of pixels in center row of two notes
        # greater than 5 * line_width, then notes are
        # beamed
        
        for j in range(len(staff_primitives)):
            if (j+1 < len(staff_primitives)
                and staff_primitives[j].getPrimitive() == "note"
                and staff_primitives[j+1].getPrimitive() == "note"
                and (staff_primitives[j].getDuration() == 1 or staff_primitives[j].getDuration() == 0.5)
                and staff_primitives[j+1].getDuration() == 1):

                # Notes of interest
                note_1_center_x = staff_primitives[j].getBox().getCenter()[0]
                note_2_center_x = staff_primitives[j+1].getBox().getCenter()[0]

                # Regular number of black pixels in staff column
                num_black_pixels = 5 * staffs[i].getLineWidth()

                # Actual number of black pixels in mid column
                center_column = (note_2_center_x - note_1_center_x) // 2
                mid_col = staff_img[:, int(note_1_center_x + center_column)]
                num_black_pixels_mid = len(np.where(mid_col == 0)[0])

                if (num_black_pixels_mid > num_black_pixels):
                    # Notes beamed
                    # Make eighth note length
                    staff_primitives[j].setDuration(0.5)
                    staff_primitives[j+1].setDuration(0.5)
                    print("[INFO] Primitive {} and {} were eighth notes misclassified as quarter notes".format(j+1, j+2))
        
        
        # ------- Account for Key Signature -------
        print("[INFO] Applying key signature note value changes")
        num_sharps = 0
        num_flats = 0
        j = 0
        while (staff_primitives[j].getDuration() == 0):
            accidental = staff_primitives[j].getPrimitive()
            if (accidental == "sharp"):
                num_sharps += 1
                j += 1

            elif (accidental == "flat"):
                num_flats += 1
                j += 1

        # Check if last accidental belongs to note

        if (j != 0):
            # Determine if accidental coupled with first note
            # Center of accidental should be within a note width from note
            max_accidental_offset_x = staff_primitives[j].getBox().getCenter()[0] - staff_primitives[j].getBox().getWidth()
            accidental_center_x = staff_primitives[j-1].getBox().getCenter()[0]
            accidental_type = staff_primitives[j-1].getPrimitive()

            if (accidental_center_x > max_accidental_offset_x):
                print("[INFO] Last accidental belongs to first note")
                num_sharps = num_sharps - 1 if accidental_type == "sharp" else num_sharps
                num_flats = num_flats - 1 if accidental_type == "flat" else num_flats

            # Modify notes in staff
            notes_to_modify = []
            if (accidental_type == "sharp"):
                print("[INFO] Key signature has {} sharp accidentals: ".format(num_sharps))
                notes_to_modify = key_signature_changes[accidental_type][num_sharps]
                # Remove accidentals from primitive list
                staff_primitives = staff_primitives[num_sharps:]
            else:
                print("[INFO] Key signature has {} flat accidentals: ".format(num_flats))
                notes_to_modify = key_signature_changes[accidental_type][num_flats]
                # Remove accidentals from primitive list
                staff_primitives = staff_primitives[num_flats:]

            print("[INFO] Corrected note values after key signature: ")
            for primitive in staff_primitives:
                type = primitive.getPrimitive()
                note = primitive.getPitch()
                if (type == "note" and note[0] in notes_to_modify):
                    new_note = MIDI_to_pitch[pitch_to_MIDI[note] + 1] if accidental_type == "sharp" else MIDI_to_pitch[pitch_to_MIDI[note] - 1]
                    primitive.setPitch(new_note)

                if (primitive.getPrimitive() == "note"):
                    print(primitive.getPitch(), end=", ")
                else:
                    print(primitive.getPrimitive(), end=", ")

            print("\n")

        # ------- Apply Sharps and Flats -------
        print("[INFO] Applying any accidental to neighboring note")
        primitive_indices_to_remove = set()
        for j in range(len(staff_primitives)):
            accidental_type = staff_primitives[j].getPrimitive()

            if (accidental_type == "flat" or accidental_type == "sharp"):
                max_accidental_offset_x = staff_primitives[j+1].getBox().getCenter()[0] - staff_primitives[j+1].getBox().getWidth()
                accidental_center_x = staff_primitives[j].getBox().getCenter()[0]
                primitive_type = staff_primitives[j+1].getPrimitive()

                if (accidental_center_x > max_accidental_offset_x and primitive_type == "note"):
                    print("Primitive has accidental associated with it")
                    note = staff_primitives[j+1].getPitch()
                    new_note = MIDI_to_pitch[pitch_to_MIDI[note] + 1] if accidental_type == "sharp" else MIDI_to_pitch[pitch_to_MIDI[note] - 1]
                    staff_primitives[j+1].setPitch(new_note)
                    primitive_indices_to_remove.add(i)

        # Removed actioned accidentals
    #     for j in primitive_indices_to_remove:
    #         del staff_primitives[j]
        staff_primitives = [staff_primitives[j] for j in range(len(staff_primitives)) if j not in primitive_indices_to_remove]

        print("[INFO] Corrected note values after accidentals: ")
        for j in range(len(staff_primitives)):
            if (staff_primitives[j].getPrimitive() == "note"):
                print(staff_primitives[j].getPitch(), end=", ")
            else:
                print(staff_primitives[j].getPrimitive(), end=", ")

        print("\n")


        # ------- Assemble Staff -------

        print("[INFO] Assembling current staff")
        bar = Bar()
        while (len(staff_primitives) > 0):
            primitive = staff_primitives.pop(0)

            if (primitive.getPrimitive() != "line"):
                bar.addPrimitive(primitive)
            else:
                staffs[i].addBar(bar)
                bar = Bar()
        # Add final bar in staff
        staffs[i].addBar(bar)
        

    print("[INFO] Sequencing MIDI")
    midi = MIDIFile(1)  # create a MIDIFile Object named (midi), numTracks = 1: only has 1 track
    track = 0
    time = 0
    channel = 0
    volume = 100

    midi.addTrackName(track, time, "Track")   # assinge the name track to the track, time (in beats) at which the track name event is placed
    midi.addTempo(track, time, 110)    # add a tempo to a track (in beats per min)


    # -------------------------------------------------------------------------------
    # Sequence MIDI
    # -------------------------------------------------------------------------------

    for i in range(len(staffs)):
        print("==== Staff {} ====".format(i+1))
        bars = staffs[i].getBars()
        for j in range(len(bars)):
            print("--- Bar {} ---".format(j + 1))
            primitives = bars[j].getPrimitives()
            for k in range(len(primitives)):
                duration = primitives[k].getDuration()
                if (primitives[k].getPrimitive() == "note"):
                    
                    print(primitives[k].getPitch())
                    
                    pitch = pitch_to_MIDI[primitives[k].getPitch()]
                    midi.addNote(track, channel, pitch, time, duration, volume)
                print(primitives[k].getPrimitive())
                print(primitives[k].getPitch())
                print(primitives[k].getDuration())
                print("-----")
                time += duration

    # ------- Write to disk -------
    print("[INFO] Writing MIDI to disk")
    binfile = open(output_path + "sound_output.mid", 'wb')
    midi.writeFile(binfile)
    binfile.close()