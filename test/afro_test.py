import afro

a = afro.cam.Streamer(config_file="/etc/afro/default")
a.initialise()
a.tx_start();

a.tx_stop();
