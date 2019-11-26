import csv

with open("train-balanced-sarcasm.csv", mode='r') as csv_file:
    with open("comments.txt", mode='w') as comment_text_file:
        with open("parents.txt", mode='w') as parent_text_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                elif row["label"] == '1':
                    comment_text_file.write(row["comment"] + "\n")
                    parent_text_file.write(row["parent_comment"] + "\n")
                    line_count += 1
            print(line_count)
    
#def onlySarcasticCommentsAndParents():
#
