import json
import numpy as np

def load_classes(): 
    return json.load(open('courses_normalized.json'))

def classes_json_to_numpy(classes_json):
    classes_np = [["Id", "Subject", "Name", "Desc", "Profs", "Days", "StartTime", "EndTime", "HasSection"]]
    
    print(classes_json[39]["StartTime"])

    for i in range(8196):
        try:
            id = classes_json[i]["Id"]
            subject = classes_json[i]["Subject"]         
            class_title = classes_json[i]["Name"]
            class_desc = classes_json[i]["Desc"]
            profs = classes_json[i]["Profs"]
            days = classes_json[i]["Days"]
            start_time = classes_json[i]["StartTime"]       
            end_time = classes_json[i]["EndTime"]
            has_section = classes_json[i]["HasSection"]

            classes_np.append([id, subject, class_title, class_desc, profs, days, start_time, end_time, has_section])
        except KeyError:
            pass

    print (np.array(classes_np))
        
if __name__ == "__main__": 
    classes_json = load_classes()
    classes_json_to_numpy(classes_json)
    



 
