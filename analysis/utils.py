import pandas as pd
import pymysql


def query_labels():
    # SQLALCHEMY_DATABASE_URI="mysql://:@:/labeler"
    conn = pymysql.connect(
        host="localhost",
        port=int(3306),
        user="root",
        passwd="mechanick244",
        db="labeler",
        charset='utf8mb4')

    # %%

    ## find and concat all tabels named *_label
    df = pd.read_sql_query("show tables;", conn)  # get name of all tables
    dfs = []
    DB_names = []
    for DB_name in df.Tables_in_labeler:
        if '_label' in DB_name:
            df = pd.read_sql_query("SELECT * FROM " + DB_name, conn)
            DB_names.append(DB_name)
            dfs.append(df)

    # inser DB names into the table
    labels = pd.concat(dfs, keys=DB_names)
    labels['db_name'] = [row.name[0] for _, row in labels.iterrows()]

    # left-join image paths
    images_info = pd.read_sql_query("SELECT * FROM cool_data", conn)
    labels = pd.merge(labels, images_info[['id', 'item_path']],
                      how='left', left_on='item_id', right_on='id')
    labels['id'] = labels['id_x']

    # left-join norming data
    norming = pd.read_csv('data/norming.csv', header=4)
    labels['Target'] = ["-".join(s.split('/')[-1].split('.')[0].split('-')[1:3])
                        for s in labels.item_path]
    labels = pd.merge(labels, norming,
                      how='left', on='Target')

    ## get users data
    users = pd.read_sql_query("SELECT * FROM labeler_user", conn)
    labels = labels.merge(right=users[["id", "email", "name"]],
                          left_on="user_id", right_on="id",
                          suffixes=("", "_user"))
    return labels


def get_img_dir(img_path):
    cfd_dir = '/Users/amirjoudaki/Downloads/CFD_v2.3/Images/'
    if is_running_local:
        img_path = cfd_dir + img_path.split('/')[-1].split('.')[0] + '.jpg'
    return img_path


def get_labels(th : int):
    if is_running_local:
        labels = pd.read_pickle("data/labels.pkl")
    else:
        labels = query_labels()
    counts = labels.user_id.value_counts()
    counts = counts[counts>=th]
    labels = labels[labels.user_id.isin(counts.index)]
    return labels


is_running_local = False