
import csv
import os


files = ["train", "test"]
for file in files:
    if not os.path.exists(f"data/{file}.csv"):
        with open(f"data/{file}.csv", "at") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["label", "title", "description"])
            with open(f"data/{file}.ft.txt", "rt") as text_file:
                for line in text_file:
                    review_data = line.split(sep=" ", maxsplit=1)
                    label = review_data[0] if len(review_data) > 1 else None
                    review_data = review_data[1].split(":", maxsplit=1)
                    title = review_data[0] if len(review_data) > 1 else None
                    review = review_data[1] if len(review_data) > 1 else None

                    writer.writerow([label, title, review])