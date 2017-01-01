import sqlite3
 
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 
# If it is not possible or desirable to put the notice in a particular
# file, then You may include the notice in a location (such as a LICENSE
# file in a relevant directory) where a recipient would be likely to look
# for such a notice.
 
# You may add additional accurate notices of copyright ownership.
 
# Written by Stefano Palmieri in December 2016
 
# This class stores unique genes in a database for the purpose of keeping track
# of innovation numbers. The fields in the database are "type", "origin" and
# "destination. In this NEAT implementation, both node and link genes are
# assigned innovation numbers. For the node genes, the origin field corresponds to
# the link or node from which the node originated from and the destination field is
# unused. For link genes, the origin field corresponds to the starting node and the
# destination field corresponds to the node where the link ends. Finally, the rowid
# field is used as the innovation number itself.
 
 
class InnovationDB(object):
 
    def __init__(self):
 
        self.conn = sqlite3.connect(":memory:")
        self.cursor = self.conn.cursor()
 
        # rowid in the innovations table signifies the innovation number for that gene
        self.cursor.execute('''CREATE TABLE innovations
                   (type TEXT NOT NULL,
                   origin INTEGER NOT NULL,
                   destination INTEGER NOT NULL,
                   PRIMARY KEY (type, origin, destination));''')
 
        print("Innovation table created successfully")
 
    def close(self):
        self.conn.close()

    # This is for directly inserting a gene into thee innovation table. Should be used for initial genes only
    def direct_insert(self, gene, origin, destination):
        self.cursor.execute("INSERT INTO innovations (type,origin,destination) VALUES (?, ?, ?)",
                            (gene, origin, destination))

 
    def retrieve_innovation_num(self, gene, origin, destination):
 
        self.cursor.execute("SELECT rowid FROM innovations WHERE type=? AND origin=? AND destination=?",
                            (gene, origin, destination))
 
        results = self.cursor.fetchall()
 
        # If the gene doesn't exist in the innovations table
        if not results:
            # add the gene to the innovation table and assign an innovation number
            self.cursor.execute("INSERT INTO innovations (type,origin,destination) VALUES (?, ?, ?)",
                                (gene, origin, destination))
            return self.cursor.lastrowid
        else:
            for row in results:
                return row[0]
 

