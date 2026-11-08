# Task 6.1 Docker Definitions

## Image

An image is a read-only package containing the filesystem, executable programs, and configuration needed to start software. `nginx:alpine` was pulled and inspected in `6.1_01_pull_and_images.txt`, while the session-pulled `alpine` image used for the named-volume demo is removed in `6.1_06_list_remove.txt`.

## Container

A container is a runnable, isolated instance created from an image, with its own process and runtime settings. The `vg-warmup` container created from the Nginx image appears in `6.1_02_run_port_volume.txt`, is entered with `docker exec` in `6.1_05_shell.txt`, and is stopped and removed in `6.1_06_list_remove.txt`.

## Layer

A layer is one cached filesystem change that contributes to an image, such as a copied entrypoint script or installed software. The `docker image history` output in `6.1_01_pull_and_images.txt` lists those changes and their sizes, showing that an image is assembled from multiple layers rather than one opaque file.

## Volume

A volume is storage managed separately from a container so data can remain available after a particular container exits. `6.1_03_volume_live_edit.txt` creates `vg-warmup-data`, writes `persisted` from one short-lived container, and reads it from another; the same transcript also shows the bind-mounted host page changing live.

## Port mapping

Port mapping publishes a container port on a host port so software outside the container can reach it. The `-p 8081:80` option and `docker port` output in `6.1_02_run_port_volume.txt` map host port 8081 to Nginx port 80, which is confirmed by the host `curl` response containing the custom page.
