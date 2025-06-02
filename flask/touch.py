from flask import Flask,jsonify,request
import matplotlib.pyplot as plt
import mysql.connector
from datetime import datetime
import pandas as pd
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

def connect_db():
    try:
        conn = mysql.connector.connect(host="localhost",user="root", password="", database="csucc-qrconnect")
        print("Database Connection Success!")
        return conn
    except mysql.connector.Error as err:
        if err.errno == err.errorcode.ER_ACCESS_DENIED_ERROR:
            print("Account credentials not correct")
        elif err.errno == err.errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    print("Something went wrong, connection not returned")
    return None

def get_attendee_records():
    conn = connect_db()
    cursor = conn.cursor()
    # query = "SELECT * FROM attendee_records WHERE event_id = 21 AND single_signin IS NULL AND check_out IS NOT NULL AND check_in IS NOT NULL"
    query = "SELECT * FROM attendee_records WHERE event_id = 37"
    results = pd.read_sql(query,conn) # this exceutes the query to fetch data and turns it into pandas Dataframe
    cursor.close
    conn.close()
    return results


##incase turn timezone to manila
# # Convert check_in and check_out timestamp string (YYYY-MM-DD HH:MM:SS) to minutes from midnight
# def data_prep(df):
#     # Localize and convert to Manila time
#     df['check_in'] = pd.to_datetime(df['check_in']).dt.tz_localize('UTC').dt.tz_convert('Asia/Manila')
#     df['check_out'] = pd.to_datetime(df['check_out']).dt.tz_localize('UTC').dt.tz_convert('Asia/Manila')
#     df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize('UTC').dt.tz_convert('Asia/Manila')

#     # Calculate minutes from midnight (in Manila time)
#     df['check_in_minutes'] = df['check_in'].dt.hour * 60 + df['check_in'].dt.minute
#     df['check_out_minutes'] = df['check_out'].dt.hour * 60 + df['check_out'].dt.minute

#     # Format the datetime fields to strings with timezone indicated
#     df['check_in'] = df['check_in'].dt.strftime('%a, %d %b %Y %H:%M:%S Manila')
#     df['check_out'] = df['check_out'].dt.strftime('%a, %d %b %Y %H:%M:%S Manila')
#     df['created_at'] = df['created_at'].dt.strftime('%a, %d %b %Y %H:%M:%S Manila')

#     return df


# Convert check_in and check_out timestamp string (YYYY-MM-DD HH:MM:SS) to minutes from midnight
def data_prep(df):
    # # Convert strings to datetime
    # df['check_in'] = pd.to_datetime(df['check_in'])
    # df['check_out'] = pd.to_datetime(df['check_out'])

    # # Then extract minutes from midnight
    # df['check_in_minutes'] = df['check_in'].dt.hour * 60 + df['check_in'].dt.minute
    # df['check_out_minutes'] = df['check_out'].dt.hour * 60 + df['check_out'].dt.minute

    df['single_signin'] = pd.to_datetime(df['single_signin'])
    df['single_signin_minutes'] = df['single_signin'].dt.hour * 60 + df['single_signin'].dt.minute
    return df



def start_anomaly_detection():
    prepData = data_prep(get_attendee_records())
    selected_features = prepData[['single_signin']]
    model = IsolationForest(n_estimators=50,contamination=0.01,max_features=1,random_state=42)
    model.fit(selected_features)
    prepData['anomaly_scores'] = model.decision_function(selected_features)
    prepData['anomaly'] = model.predict(selected_features)



    plt.scatter(
        prepData['single_signin_minutes'],
        prepData['single_signin'],
        c=prepData['anomaly'],
        cmap='coolwarm'
    )

    plt.xlabel('Single Signin Minutes')
    plt.ylabel('Attendance Sign in')
    plt.title('Anomalies in Attendance Data')
    plt.show()

    # plt.scatter(
    #     prepData['check_in_minutes'],
    #     prepData['check_out_minutes'],
    #     c=prepData['anomaly'],
    #     cmap='coolwarm'
    # )
    # plt.xlabel('Check-in Minutes')
    # plt.ylabel('Check-out Minutes')
    # plt.title('Anomalies in Attendance Data')
    # plt.show()

    return prepData


@app.route('/test', methods=['GET'])
def test_route ():
    data = start_anomaly_detection()
    return jsonify(data.to_dict(orient="records")) #convert each row in dataframe to dictionaries with key value pairs

if __name__ == '__main__' :
    app.run(debug=True)


#     from flask import Flask, request, jsonify
# import mysql.connector
# from sklearn.ensemble import IsolationForest

# app = Flask(__name__)

# # 🧠 Train Isolation Forest with sample data (replace with actual training data)
# model = IsolationForest()
# model.fit([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

# # 🔐 MySQL connection function (adjust based on your XAMPP setup)
# def get_db_connection():
#     return mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password="",  # Default XAMPP password
#         database="your_database"
#     )

# # 📥 Function to fetch data from MySQL
# def fetch_user_data(user_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT feature1, feature2, feature3 FROM user_activity WHERE id = %s", (user_id,))
#     result = cursor.fetchone()
#     cursor.close()
#     conn.close()
#     if result:
#         return list(result)
#     return None

# # ⚙️ Function to run Isolation Forest prediction
# def detect_anomaly(features):
#     return model.predict([features])[0]  # -1 = anomaly, 1 = normal

# # 🚀 Flask route
# @app.route('/detect-anomaly', methods=['POST'])
# def detect_anomaly_route():
#     data = request.get_json()
#     user_id = data.get('user_id')

#     if user_id is None:
#         return jsonify({"error": "Missing user_id"}), 400

#     user_data = fetch_user_data(user_id)
#     if user_data is None:
#         return jsonify({"error": "User data not found"}), 404

#     prediction = detect_anomaly(user_data)

#     return jsonify({
#         "user_id": user_id,
#         "data": user_data,
#         "anomaly": prediction == -1
#     })

# if __name__ == '__main__':
#     app.run(debug=True)




# Convert check_in and check_out timestamp string (YYYY-MM-DD HH:MM:SS) to minutes from midnight
# def data_prep(attendee_records):
#     converted_results = []

#     for row in attendee_records:
#         # check_in = datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S")
#         # check_out = datetime.strptime(row[4], "%Y-%m-%d %H:%M:%S")

#         print(row[3])

#         # if row[3] is not None:
#         #     check_in_mins = (row[3].hour * 60) + row[3].minute

#         # if row[4] is not None:
#         #     check_out_mins = (row[4].hour * 60) + row[4].minute

#         check_in_mins = (row[3].hour * 60) + row[3].minute
#         check_out_mins = (row[4].hour * 60) + row[4].minute

#         # check_in_mins = (check_in.hour * 60) + check_in.minute
#         # check_out_mins = (check_out.hour * 60) + check_out.minute

#         converted_results.append([
#             row[0],
#             row[1],
#             row[2],
#             check_in_mins,
#             check_out_mins,
#             row[5],
#             row[6],
#             row[7],
#         ])

#     return converted_results

