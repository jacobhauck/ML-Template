import sqlite3
import os
import csv


def as_sql(value):
    if isinstance(value, str):
        return f"'{value}'"
    else:
        return str(value)


class DatasetLibrary:
    def __init__(self, name, ext):
        self.root = os.path.join('data', name)
        self.ext = ext
        os.makedirs(self.root, exist_ok=True)
        self.db = sqlite3.connect(os.path.join(self.root, 'library.db'))
        tables = self.db.execute("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        if ('datasets',) not in tables:
            self.db.cursor().execute('''
                CREATE TABLE datasets (
                    id INTEGER PRIMARY KEY
                )
            ''')
            self.db.commit()
    
    def dataset_path(self, split, dataset_id):
        return os.path.join(self.root, f'{split}-{dataset_id}.{self.ext}')
    
    def splits(self, dataset_id):
        found = []
        for item in os.listdir(self.root):
            try:
                clean = item[:-len(self.ext) - 1]
                check_id = int(clean.split('-')[-1])
                if check_id == dataset_id:
                    found.append(clean[:clean.rfind('-')])
            except Exception:
                pass
        
        return found
    
    def add_property(self, name, dtype=float):
        if dtype not in (int, float, str):
            raise ValueError('Invalid dtype; choose from int, float or str.')

        cur = self.db.cursor()
        sql_type = {int: 'INTEGER', float: 'REAL', str: 'TEXT'}[dtype]
        cur.execute(f'''
            ALTER TABLE datasets ADD COLUMN {name} {sql_type}
        ''')
        self.db.commit()

    def create_dataset(self, **properties):
        try:
            cur = self.db.cursor()
            existing_props = set(row[1] for row in cur.execute('PRAGMA table_info(datasets)').fetchall())
            for name, value in properties.items():
                if name not in existing_props:
                    self.add_property(name, dtype=type(value))

            sql = 'INSERT INTO datasets ('
            sql += ', '.join(name for name in properties)
            sql += ') VALUES ('
            sql += ', '.join('?' * len(properties))
            sql += ')'
            dataset_id = cur.execute(sql, tuple(properties.values())).lastrowid
            self.db.commit()

        except Exception as e:
            print('Error occurred attempting to create dataset!')
            raise e

        return dataset_id
    
    def delete_dataset(self, dataset_id):
        try:
            cur = self.db.cursor()
            cur.execute('DELETE FROM datasets WHERE id=?', (dataset_id,))
            for split in self.splits(dataset_id):
                print(f'Removing split {split} of dataset {dataset_id}')
                os.remove(os.path.join(self.root, f'{split}-{dataset_id}.{self.ext}'))
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            print('Failed to delete dataset!')
            raise e
    
    def export(self, file_name):
        output = [list(row[1] for row in self.db.execute('PRAGMA table_info(datasets)').fetchall())]
        for row in self.db.execute('SELECT * FROM datasets').fetchall():
            output.append(row)

        with open(file_name, 'w', newline='') as f:
            csv.writer(f).writerows(output)
