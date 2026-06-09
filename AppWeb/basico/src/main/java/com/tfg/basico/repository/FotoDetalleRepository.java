package com.tfg.basico.repository;

import com.tfg.basico.model.FotoDetalle;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

/**
 * Interfaz para gestionar las consultas de la tabla "foto_detalle".
 */
@Repository
public interface FotoDetalleRepository extends JpaRepository<FotoDetalle, Integer> {
    
    // Pregunta a la base de datos si ya existe guardada una foto marcada con ese nombre
    boolean existsByRutaImagenMarcada(String rutaImagenMarcada);
    
    // Pregunta a la base de datos si ya existe una foto original con ese nombre
    boolean existsByRutaImagenOriginal(String rutaImagenOriginal);
}