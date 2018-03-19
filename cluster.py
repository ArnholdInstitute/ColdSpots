#!/usr/bin/env python

'''
Cluster buildings together for a particular country/project.
'''

import pdb, json, os, argparse, numpy as np
from shapely.geometry import shape, MultiPoint, mapping, MultiPolygon
from sklearn.cluster import DBSCAN
from db import aigh_conn, atlas_conn
from datetime import datetime

def get_srid(geom, version):
    '''
    The PostGIS function for DBSCAN clustering operates on euclidean 
    distance.  We'd like to specify the distance threshold (epsilon)
    in terms of meters, so we'll convert the geometries to a meter based
    geometry so that euclidean distance will work. 

    Arguments:
        geom : shapely.Geometry - Geometry of the catchment zone
        version : UUID - Version of the buildings we are clustering
    Returns:
        int - This is the SRID of the projection we should transform to
    '''
    geom = shape(geom)
    with aigh_conn.cursor() as cur:
        cur.execute("""
            SELECT _ST_BestSRID(ST_Centroid(ST_ConvexHull(ST_Collect(geom))))
            FROM buildings.buildings 
            WHERE 
                version=%s AND 
                ST_Contains(ST_GeomFromText(%s, 4326), geom);
        """, (version, geom.wkt))
        return cur.fetchone()[0]

def get_latest_version(country):
    '''
    Get the most recent version number for a particular country.  This provides
    a reasonable default if the user doesn't specify a version.

    Arguments:
        country : text - The country that we are checking versions for
    Returns:
        UUID - The version number
    '''
    with aigh_conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (ts) version 
            FROM buildings.buildings 
            WHERE project=%s ORDER BY ts LIMIT 1
        """, (country, ))
        return cur.fetchone()[0]

def transfer(region, version):
    geom = shape(region['geometry'])
    project = region['properties']['project']
    org_id = region['properties']['organization_id']

    TS = datetime.now()

    with aigh_conn.cursor() as aigh_cur, atlas_conn.cursor() as atlas_cur:
        print('Transfering clusters...')
        aigh_cur.execute("""
            SELECT
                version,
                %s as organization_id,
                size as building_count,
                geom,
                ST_Centroid(geom) as centroid,
                site_dist as site_distance
            FROM clusters
            WHERE clusters.version=%s AND ST_Relate(ST_GeomFromText(%s, 4326), geom, '2********')
        """, (org_id, version, geom.wkt))

        atlas_cur.execute("""
            DELETE FROM buildings 
            WHERE version=%s AND 
                ST_Relate(ST_GeomFromText(%s, 4326), geom, '2********');
            DELETE FROM building_clusters WHERE organization_id=%s AND version=%s;
            DELETE FROM active_building_clusters WHERE organization_id=%s AND version=%s;
        """, (version, geom.wkt, org_id, version, org_id, version))

        atlas_cur.execute("""
            UPDATE active_building_clusters SET active=false WHERE organization_id=%s;
            INSERT INTO active_building_clusters (version, organization_id, active, entered)
            VALUES (%s, %s, true, %s)
        """, (org_id, version, org_id, TS))

        args_str = ','.join(atlas_cur.mogrify("(%s,%s,%s,%s,%s,%s)", x) for x in aigh_cur)
        atlas_cur.execute("""
            INSERT INTO building_clusters (
                version, 
                organization_id, 
                building_count, 
                geom, 
                centroid, 
                site_distance
            ) VALUES %s
        """ % args_str)

    with aigh_conn.cursor(name='aigh') as aigh_cur, atlas_conn.cursor() as atlas_cur:
        print('Transfering buildings...')
        aigh_cur.execute("""
            SELECT geom, %s as version
            FROM buildings.buildings as b
            WHERE version=%s AND ST_Relate(ST_GeomFromText(%s, 4326), geom, '2********')
        """, (version, version, geom.wkt))

        count = 0
        while True:
            rows = aigh_cur.fetchmany(2048)
            if len(rows) == 0:
                break
            args_str = ','.join(atlas_cur.mogrify("(%s,%s)", x) for x in rows)
            atlas_cur.execute("INSERT INTO buildings (geom, version) VALUES %s" % args_str)
            count += len(rows)
            print('Inserted %d rows' % count)
    atlas_conn.commit()


def cluster(region, version, epsilon):
    '''
    Cluster buildings together for a given region and insert them into the clusters table
    Arguments:
        region : GeoJSON (Feature) - GeoJSON object describing the geometry of the 
            catchment zone for the organization we are clustering for.  The properties
            field must contain a "project" field indicating which project in the 
            bulidings.buildings table that it belongs to.
        version : UUID - Version number of the model used to predict the buildings
        epsilon : float - Minimum distance for a point to be considered part of a cluster
    '''

    geom = shape(region['geometry'])
    project = region['properties']['project']

    srid = get_srid(geom, version)

    with aigh_conn.cursor() as cur:
        # Create the table or empty out any clusters with the same version ID
        cur.execute("""
            CREATE TABLE IF NOT EXISTS clusters(
                id serial, 
                project text, 
                size int, 
                geom geometry('geometry', 4326), 
                version uuid
            );
            DELETE FROM clusters WHERE ST_Relate(ST_GeomFromText(%s, 4326), geom, '2********') AND version=%s
        """, (geom.wkt, version))

        print('Clustering buildings...')
        cur.execute("""
            INSERT INTO clusters (project, size, geom, version)
            SELECT
                %s as project,
                COUNT(*) as size,
                ST_ConvexHull(ST_Collect(geom)) as geom,
                %s as version
            FROM (
                SELECT
                    ST_ClusterDBSCAN(ST_Transform(geom, %s), eps := %s, minpoints := 3) over () as cid,
                    geom
                FROM buildings.buildings
                WHERE "version"=%s AND ST_Contains(ST_GeomFromText(%s, 4326), geom)
            )clustering
            GROUP BY cid
        """, (project, version, srid, epsilon, version, geom.wkt))

        print('Computing nearest fixtures for each cluster...')
        cur.execute("""
            UPDATE clusters SET site_dist=dist FROM(
                SELECT DISTINCT ON (clusters.id)
                    clusters.id as cluster_id,
                    ST_Distance(clusters.geom::geography, fixtures.geom::geography) as dist
                FROM clusters, fixtures
                WHERE clusters.version=%s AND ST_Relate(ST_GeomFromText(%s, 4326), geom, '2********')
                ORDER BY clusters.id, dist
            )q WHERE id=cluster_id;
        """, (version, geom.wkt))

    aigh_conn.commit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', default=500, type=float, help='Max cluster distance (in meters)')
    parser.add_argument('--version', default=None, help='Version number to cluster (defaults to latest timestamp)')
    parser.add_argument('--region', required=True, help='Region in which to cluster buildings together')
    args = parser.parse_args()

    region = json.load(open(args.region))
    args.version = args.version if args.version else get_latest_version(region['properties']['project'])

    cluster(region, args.version, args.epsilon)
    transfer(region, args.version)
