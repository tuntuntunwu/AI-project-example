from re import search
from pyodbc import connect


def singleInfo(img_path):
    # connect to .mdb
    MDB = "./res/新金文库.mdb"
    DRV = "{Microsoft Access Driver (*.mdb, *.accdb)}"
    con = connect('DRIVER={};DBQ={};'.format(DRV, MDB))
    cur = con.cursor()

    search_name_result = search(r"\..*?[/\\]", img_path[::-1]).group()[::-1]
    if not search_name_result:
        clear(cur, con)
        return None
    search_name = search_name_result[1:-1]
    # print(search_name)

    SQL = "SELECT 新统计用字头,字形出处,器名,释文表.时代,释文新\
        FROM 单字字形表,释文表 WHERE 册数_器名ID_字头ID_重铭 = '"\
        + search_name + "' AND 单字字形表.器名ID=释文表.器名ID;"
    # SELECT 新统计用字头,字形出处,器名,释文表.时代,释文新 FROM 单字字形表,释文表 WHERE 册数_器名ID_字头ID_重铭 = '01_00001_001_A' AND 单字字形表.器名ID=释文表.器名ID;
    rows = cur.execute(SQL).fetchone()
    if rows:
        clear(cur, con)
        return rows
    else:
        clear(cur, con)
        return None


def rubbingInfo(img_path):
    # connect to .mdb
    MDB = "./res/新金文库.mdb"
    DRV = "{Microsoft Access Driver (*.mdb, *.accdb)}"
    con = connect('DRIVER={};DBQ={};'.format(DRV, MDB))
    cur = con.cursor()

    result_obj = search(r"\.(\d*)[\/\\].*?(\d*)[\/\\]", img_path[::-1])
    if not result_obj:
        clear(cur, con)
        return None
    else:
        pa0 = int(result_obj.group(1)[::-1])
        pa1 = int(result_obj.group(2)[::-1])

        SQL = "SELECT 释文1 FROM 释文表 WHERE 册数=" + \
            str(pa1) + " AND 第一出处='集成" + str(pa0) + "';"
        # print(SQL)
        rows = cur.execute(SQL).fetchone()
        if rows:
            clear(cur, con)
            return rows
        else:
            clear(cur, con)
            return None


def clear(cur, con):
    # close .mdb
    cur.close()
    con.close()


if __name__ == '__main__':
    pass
