import myfunctions as fn
import sys
def main():
    db = fn.Database(sys.argv[1])
    IDs = db.getIDlist()
    for ID in IDs[1:]:
        print(ID)

if __name__ == "__main__":
    main()
